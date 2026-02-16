"""
STEP 6 — Adaptive Multi-Stage Constrained Reasoning Layer
Production hardened: structured JSON outputs + repair + deterministic fallback.
Replaces prior regex-only numeric banning with schema enforcement and fallback.
"""

import logging
import re
import json
import math
from typing import Dict, Any, List, Optional, Union, cast
from pathlib import Path

ALLOWED_ROIS = [
    "hippocampus",
    "entorhinal_cortex",
    "temporal_lobe",
    "parietal_lobe",
    "ventricles"
]

FORBIDDEN_TERMS = [
    "alzheimer",
    "dementia",
    "mci",
    "mild cognitive impairment",
    "diagnosis",
    "will develop"
]

MISMATCH_TERMS = [
    "discrepancy",
    "inconsistent",
    "contradict",
    "mismatch",
    "does not align",
    "spatial conflict",
    "anatomical divergence"
]

MAX_RETRY = 2
NUMERIC_TOLERANCE = 0.05

FINAL_SCHEMA_KEYS = {
    "roi_interpretations",
    "final_narrative",
    "classification",
    "confidence_level",
    "warning_flag",
    "repair_attempted",
    "fallback_used"
}


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------
def setup_reasoning_logger(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "step6_reasoning.log"
    logger = logging.getLogger("ReasoningLayer")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------
# Basic validators
# ---------------------------------------------------------
def _validate_payload(payload: Dict[str, Any]) -> None:
    required = ["metadata", "roi_metrics", "progression", "context_images"]
    for k in required:
        if k not in payload:
            raise ValueError(f"Missing payload key: {k}")


def _check_forbidden_terms(text: str) -> None:
    lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if term in lower:
            raise ValueError(f"Forbidden medical claim detected: {term}")


def _allowed_numeric_values(payload: Dict[str, Any]) -> List[float]:
    allowed = []
    for roi in ALLOWED_ROIS:
        allowed.append(float(payload["roi_metrics"][roi]["annual_percent_change"]))
        allowed.append(float(payload["roi_metrics"][roi]["z_score"]))
    allowed.append(float(payload["progression"].get("score", 0)))
    allowed.append(float(payload["metadata"].get("interval_years", 0.0)))
    
    # Include numbers found in progression rationale (thresholds)
    for item in payload["progression"].get("rationale", []):
        for m in re.findall(r"-?\d+\.?\d+", item):
            try:
                allowed.append(float(m))
            except:
                pass
    return allowed


def _extract_numbers_from_jsonish(obj: Any) -> List[float]:
    nums = []
    if isinstance(obj, dict):
        for v in obj.values():
            nums += _extract_numbers_from_jsonish(v)
    elif isinstance(obj, list):
        for v in obj:
            nums += _extract_numbers_from_jsonish(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        nums.append(float(obj))
    elif isinstance(obj, str):
        for m in re.findall(r"-?\d+\.?\d+", obj):
            try:
                nums.append(float(m))
            except:
                pass
    return nums


def _numbers_within_allowed(found_nums: List[float], allowed: List[float]) -> Optional[float]:
    for num in found_nums:
        if not any(abs(num - a) <= NUMERIC_TOLERANCE for a in allowed):
            return num
    return None


def _validate_final_json(final_obj: Dict[str, Any], payload: Dict[str, Any]) -> None:
    # 1) Required top-level keys
    for k in ["roi_interpretations", "final_narrative", "classification", "confidence_level"]:
        if k not in final_obj:
            raise ValueError(f"Missing top-level key in final JSON: {k}")

    # 2) ROI keys presence & numeric match
    roi_block = final_obj["roi_interpretations"]
    for roi in ALLOWED_ROIS:
        if roi not in roi_block:
            raise ValueError(f"Missing ROI in final JSON: {roi}")
        entry = roi_block[roi]
        for nk in ["annual_percent_change", "z_score"]:
            if nk not in entry:
                raise ValueError(f"Missing numeric key {nk} for ROI {roi}")
            
            payload_val = float(payload["roi_metrics"][roi]["annual_percent_change"]
                                if nk == "annual_percent_change" else payload["roi_metrics"][roi]["z_score"])
            
            if not math.isclose(float(entry[nk]), float(payload_val), rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE):
                raise ValueError(f"Numeric mismatch for {roi} {nk}: reported {entry[nk]} vs payload {payload_val}")

    # 3) Classification block consistency
    cls = final_obj["classification"]
    if "class" not in cls or "score" not in cls or "rationale" not in cls:
        raise ValueError("classification block incomplete")
    
    expected_class = payload["progression"]["class"]
    if cls["class"].lower() != expected_class.lower():
        raise ValueError("Classification text does not match deterministic classification in payload.")
    
    if int(cls["score"]) != int(payload["progression"].get("score", 0)):
        raise ValueError("Classification score mismatch.")

    # 4) Numeric integrity check for stray numbers
    allowed = _allowed_numeric_values(payload)
    found_nums = _extract_numbers_from_jsonish(final_obj)
    unauthorized = _numbers_within_allowed(found_nums, allowed)
    if unauthorized is not None:
        raise ValueError(f"Unauthorized numeric value detected: {unauthorized}")


def _extract_json_substring(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start:end+1]
    stack = []
    for ch in candidate:
        if ch == "{":
            stack.append(1)
        elif ch == "}":
            if not stack:
                return None
            stack.pop()
    if stack:
        return None
    return candidate


# ---------------------------------------------------------
# Deterministic fallback generator (no LLM)
# ---------------------------------------------------------
def deterministic_fallback(
    payload: Dict[str, Any], 
    stage1: str, 
    stage2: str, 
    stage3: str, 
    logger: logging.Logger
) -> Dict[str, Any]:
    
    logger.info("Using deterministic fallback to generate final JSON (no LLM).")

    roi_block = {}
    for roi in ALLOWED_ROIS:
        apc = round(float(payload["roi_metrics"][roi]["annual_percent_change"]), 3)
        z = round(float(payload["roi_metrics"][roi]["z_score"]), 3)
        interp = f"{roi.replace('_', ' ').title()}: annual change {apc}% per year (Z = {z})."
        roi_block[roi] = {
            "annual_percent_change": apc,
            "z_score": z,
            "interpretation": interp
        }

    classification = {
        "class": payload["progression"]["class"],
        "score": int(payload["progression"].get("score", 0)),
        "rationale": payload["progression"].get("rationale", [])
    }

    final_narrative = (
        "Deterministic fallback narrative synthesized from validated numeric payload. "
        "ROI interpretations produced programmatically to ensure numeric integrity."
    )

    return {
        "roi_interpretations": roi_block,
        "final_narrative": final_narrative,
        "classification": classification,
        "confidence_level": "Low (fallback)",
        "warning_flag": None,
        "repair_attempted": True,
        "fallback_used": True
    }


# ---------------------------------------------------------
# Core executor
# ---------------------------------------------------------
def _execute_with_retry_json(
    stage_name: str,
    prompt_package: Dict[str, Any],
    medgemma_client: Any,
    payload: Dict[str, Any],
    logger: logging.Logger,
    expect_json: bool = False,
    strict: bool = False,
    stage_trace_texts: Optional[Dict[str, Any]] = None
) -> Union[str, Dict[str, Any]]:

    base_prompt = prompt_package["text"]
    images = prompt_package.get("images", [])
    stage = prompt_package.get("stage", "stage4")

    for attempt in range(MAX_RETRY + 1):
        logger.info(f"{stage_name} — Attempt {attempt+1}")
        response = medgemma_client(prompt_package)
        
        if not isinstance(response, str):
            raise ValueError("Model response must be string.")

        logger.debug(f"{stage_name} raw output:\n{response}")

        try:
            _check_forbidden_terms(response)

            # --- TEXT MODE (Stages 1-3) ---
            if not expect_json:
                if strict:
                    for roi in ALLOWED_ROIS:
                        apc = str(payload["roi_metrics"][roi]["annual_percent_change"])
                        z = str(payload["roi_metrics"][roi]["z_score"])
                        if apc not in response or z not in response:
                            raise ValueError(f"Missing numeric value in text output for ROI {roi}.")
                logger.info(f"{stage_name} passed validation (non-JSON).")
                return response

            # --- JSON MODE (Stage 4) ---
            parsed = None
            try:
                parsed = json.loads(response)
            except Exception:
                candidate = _extract_json_substring(response)
                if candidate:
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        parsed = None

            if parsed is None:
                raise ValueError("Model output is not valid JSON.")

            _validate_final_json(parsed, payload)

            parsed["repair_attempted"] = attempt > 0
            parsed["fallback_used"] = False
            logger.info(f"{stage_name} JSON passed validation.")
            return parsed

        except Exception as e:
            logger.warning(f"{stage_name} validation failed: {e}")

            if attempt == MAX_RETRY:
                if expect_json:
                    logger.error(f"{stage_name} failed. Using deterministic fallback.")
                    s1 = str((stage_trace_texts or {}).get("stage1", ""))
                    s2 = str((stage_trace_texts or {}).get("stage2", ""))
                    s3 = str((stage_trace_texts or {}).get("stage3", ""))
                    return deterministic_fallback(payload, s1, s2, s3, logger)
                else:
                    raise e

            # Construct Repair Prompt
            corrected_prompt = (
                base_prompt
                + "\n\nVALIDATION ERROR:\n"
                + str(e)
                + "\n\nREQUIREMENTS FOR THE NEXT RESPONSE (NON-NEGOTIABLE):\n"
                + "1) Return ONLY valid JSON.\n"
                + "2) Numeric values must match the provided payload EXACTLY.\n"
                + "3) Do NOT include forbidden medical claims.\n"
            )

            prompt_package = {
                "text": corrected_prompt,
                "images": images,
                "stage": stage
            }

    raise RuntimeError("Unexpected control flow.")


# ---------------------------------------------------------
# Main entry
# ---------------------------------------------------------
def run_multistage_reasoning(
    payload: Dict[str, Any],
    prompt_builder: Any,
    medgemma_client: Any,
    output_dir: Path
) -> Dict[str, Any]:

    logger = setup_reasoning_logger(output_dir)
    logger.info("========== Starting Step 6 Reasoning ==========")

    _validate_payload(payload)

    # Stage 1 — Quantitative Interpretation (Returns String)
    stage1_pkg = prompt_builder.build_numeric_prompt(payload)
    stage1_text = cast(str, _execute_with_retry_json(
        "Stage 1 — Quantitative Interpretation",
        stage1_pkg,
        medgemma_client,
        payload,
        logger,
        expect_json=False,
        strict=False
    ))

    # Stage 2 — Multimodal Integration (Returns String)
    stage2_pkg = prompt_builder.build_multimodal_prompt(payload, stage1_text)
    stage2_text = cast(str, _execute_with_retry_json(
        "Stage 2 — Multimodal Integration",
        stage2_pkg,
        medgemma_client,
        payload,
        logger,
        expect_json=False,
        strict=False
    ))

    # Stage 3 — Self Verification (Returns String)
    stage3_pkg = prompt_builder.build_verification_prompt(payload, stage1_text, stage2_text)
    stage3_text = cast(str, _execute_with_retry_json(
        "Stage 3 — Self Verification",
        stage3_pkg,
        medgemma_client,
        payload,
        logger,
        expect_json=False,
        strict=False
    ))

    # Stage 4 — Final Clinical Narrative (Returns Dictionary)
    stage4_pkg = prompt_builder.build_simplification_prompt(payload, stage3_text)
    stage_trace_texts = {
        "stage1": stage1_text, 
        "stage2": stage2_text, 
        "stage3": stage3_text
    }
    
    final_json = cast(Dict[str, Any], _execute_with_retry_json(
        "Stage 4 — Final Clinical Narrative",
        stage4_pkg,
        medgemma_client,
        payload,
        logger,
        expect_json=True,
        strict=True,
        stage_trace_texts=stage_trace_texts
    ))

    # Consistency Check (Redundant safety check)
    if not final_json.get("fallback_used", False):
        expected_class = payload["progression"]["class"]
        # Safe access because final_json is cast to Dict
        if final_json["classification"]["class"].lower() != expected_class.lower():
            raise ValueError("Final JSON classification mismatch AFTER repair.")

    # Downgrade confidence if visual mismatch detected
    lower_s2 = stage2_text.lower()
    warning_flag = final_json.get("warning_flag", None)
    
    if any(term in lower_s2 for term in MISMATCH_TERMS) and (warning_flag is None):
        final_json["warning_flag"] = "Visual-Numeric Mismatch"
        final_json["confidence_level"] = "Low (Visual Mismatch)"

    logger.info("========== Step 6 Completed Successfully ==========")
    return {
        "thinking_trace": {
            "stage1_quantitative": stage1_text,
            "stage2_multimodal": stage2_text,
            "stage3_verification": stage3_text
        },
        "final_output": final_json,
        "confidence_level": final_json.get("confidence_level"),
        "warning_flag": final_json.get("warning_flag")
    }