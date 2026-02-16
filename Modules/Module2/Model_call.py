import os
import time
import gc
import torch
import psutil
import logging
from typing import Dict, Any, List, Optional
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer
)

MODEL_PATH = r"model_store\medgemma-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STOP_KEYWORDS = ["Disclaimer:", "Note:", "###", "<unused", "Patient Information:"]

STAGE_TOKEN_CAPS = {
    "stage1": 512,
    "stage2": 768,
    "stage3": 512,
    "stage4": 1024
}


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

def get_ram_mb():
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)


def get_gpu_stats():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "total_mb": round(props.total_memory / (1024 ** 2), 2),
    }


# ----------------------------------------------------------
# Stopping Criteria
# ----------------------------------------------------------

class NeuroStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, start_len, min_tokens=40):
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.min_tokens = min_tokens

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[-1] - self.start_len < self.min_tokens:
            return False
        text = self.tokenizer.decode(
            input_ids[0][self.start_len:], skip_special_tokens=True
        )
        return any(k in text[-50:] for k in STOP_KEYWORDS)


# ----------------------------------------------------------
# Client
# ----------------------------------------------------------

class MedGemmaClient:

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("MedGemmaClient")
        self.model, self.processor = self._load_model()
        self.streamer = TextStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        self.cached_images = None
        self._warmup()

    # ------------------------------------------------------

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        self.logger.info("Loading MedGemma (4-bit NF4)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        processor = AutoProcessor.from_pretrained(MODEL_PATH)

        model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )


        model.eval()
        return model, processor

    # ------------------------------------------------------

    def _warmup(self):
        dummy = self.processor(text="warmup", return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            _ = self.model.generate(**dummy, max_new_tokens=10)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------

    def _prepare_images(self, image_paths: List[str]):
        if not image_paths:
            return None

        if self.cached_images is None:
            self.cached_images = [
                Image.open(p).convert("RGB") for p in image_paths
            ]

        return self.cached_images

    # ------------------------------------------------------

    def generate(self, prompt_package: Dict[str, Any]) -> str:

        text = prompt_package["text"]
        stage = prompt_package.get("stage", "stage4")
        image_paths: List[str] = prompt_package.get("images", [])

        images = self._prepare_images(image_paths)

        content = []
        if images:
            for _ in images:
                content.append({"type": "image"})
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt"
        ).to(DEVICE) if images else self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(DEVICE)

        start_len = inputs["input_ids"].shape[1]

        stop_criteria = StoppingCriteriaList([
            NeuroStoppingCriteria(self.processor.tokenizer, start_len)
        ])

        max_tokens = STAGE_TOKEN_CAPS.get(stage, 768)

        budgets = [
            max_tokens,
            int(max_tokens * 0.75),
            512,
            384,
            256
        ]

        ram_pre = get_ram_mb()
        gpu_pre = get_gpu_stats()
        t0 = time.perf_counter()

        output_ids = None
        # ----------------------------------------------------------
        # FIX: Dynamic Repetition Penalty
        # ----------------------------------------------------------
        # Stage 3 (Verification) and Stage 4 (Final Narrative) must 
        # repeat exact numbers. Penalty causes "1.123" to become "1.124".
        # Disable penalty (1.0) for strict stages.
        # ----------------------------------------------------------
        current_rep_penalty = 1.0 if stage in ["stage3", "stage4"] else 1.1

        for budget in budgets:
                    try:
                        with torch.inference_mode():
                            output_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=budget,
                                do_sample=False,
                                temperature=0.0,
                                repetition_penalty=current_rep_penalty,  # <--- APPLIED HERE
                                use_cache=True,
                                streamer=self.streamer,
                                stopping_criteria=stop_criteria,
                                pad_token_id=self.processor.tokenizer.eos_token_id
                            )
                        break
                    except torch.cuda.OutOfMemoryError:
                        self.logger.warning(f"OOM at {budget} tokens. Retrying lower.")
                        torch.cuda.empty_cache()
                        gc.collect()

        if output_ids is None:
            raise RuntimeError("All OOM retries failed.")

        duration = round(time.perf_counter() - t0, 2)

        generated_tokens = output_ids[0][start_len:]
        raw_text = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        for k in STOP_KEYWORDS:
            raw_text = raw_text.split(k)[0]

        ram_post = get_ram_mb()
        gpu_post = get_gpu_stats()

        self.logger.info({
            "stage": stage,
            "duration_sec": duration,
            "tokens": len(generated_tokens),
            "ram_pre_mb": ram_pre,
            "ram_post_mb": ram_post,
            "gpu_pre": gpu_pre,
            "gpu_post": gpu_post
        })

        return raw_text.strip()
