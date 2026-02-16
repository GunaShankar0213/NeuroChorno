"""
Module 2 Orchestrator
Full deterministic quantitative pipeline.
QA-hardened, dual-atlas compliant.
Now includes Step 6 — Multi-Stage MedGemma Reasoning.
"""

from asyncio.log import logger
from pathlib import Path
import json
import traceback

from Modules.Module2.logger import setup_logger
from Modules.Module2.step1_register_atlas import register_atlas_to_subject
from Modules.Module2.step2_roi_extraction import compute_roi_metrics
from Modules.Module2.step3_zscore_engine import compute_z_scores
from Modules.Module2.step4_scoring_engine import classify_progression
from Modules.Module2.payload_builder import build_intelligence_payload
from Modules.Module2.export_t1_slice import export_t1_axial_slice

# STEP 6 IMPORTS
from Modules.Module2.Prompts import (
    build_numeric_prompt,
    build_multimodal_prompt,
    build_verification_prompt,
    build_simplification_prompt
)
from Modules.Module2.reasoning_engine import run_multistage_reasoning
from Modules.Module2.Model_call import MedGemmaClient

_MEDGEMMA_INSTANCE = None
# ==========================================================
# STATIC PATH CONFIGURATION
# ==========================================================

CONFIG_DIR = Path("Modules/Module2/config")
ATLAS_DIR = Path("module2/atlas")

ROI_CONFIG_PATH = CONFIG_DIR / "roi_labels.json"
NORMATIVE_PATH = CONFIG_DIR / "normative_reference.json"

MNI_TEMPLATE_PATH = ATLAS_DIR / "MNI152_T1_1mm.nii.gz"
CORTICAL_ATLAS_PATH = ATLAS_DIR / "harvard_oxford_cortical.nii.gz"
SUBCORTICAL_ATLAS_PATH = ATLAS_DIR / "harvard_oxford_subcortical.nii.gz"

# T1_FOLLOWUP_PATH = Path(
#     "Data/outputs/module1/02_bias_corrected/T1/T1_bet_cropped_n4.nii.gz"
# )

# JACOBIAN_OVERLAY_PATH = Path(
#     "Data/outputs/module1/jacobian_overlay.png"
# )


# ==========================================================
# MAIN ORCHESTRATION FUNCTION
# ==========================================================
def get_medgemma_client(logger):
    global _MEDGEMMA_INSTANCE
    if _MEDGEMMA_INSTANCE is None:
        logger.info("Initializing Global MedGemma Client (Singleton)...")
        _MEDGEMMA_INSTANCE = MedGemmaClient(logger)
    return _MEDGEMMA_INSTANCE

def run_module2(
    jacobian_path: Path,
    t0_path: Path,
    t1_followup_path: Path,
    jacobian_overlay_path: Path,
    age: int,
    sex: str,
    interval_days: float,
    output_dir: Path,
):

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    logger.info("========================================")
    logger.info("Starting Module 2 — Quantitative Engine")
    logger.info("========================================")

    required_paths = [
        MNI_TEMPLATE_PATH,
        CORTICAL_ATLAS_PATH,
        SUBCORTICAL_ATLAS_PATH,
        t1_followup_path,
        jacobian_overlay_path,
        jacobian_path,
        t0_path,
        ROI_CONFIG_PATH,
        NORMATIVE_PATH
    ]

    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    try:

        # ==================================================
        # STEP 1
        # ==================================================

        logger.info("STEP 1 — Registering Atlas to Subject")

        atlas_output_dir = output_dir / "step1_atlas_registration"

        step1_result = register_atlas_to_subject(
            t0_path=t0_path,
            mni_template_path=MNI_TEMPLATE_PATH,
            atlas_cortical_path=CORTICAL_ATLAS_PATH,
            atlas_subcortical_path=SUBCORTICAL_ATLAS_PATH,
            output_dir=atlas_output_dir,
            logger=logger
        )

        cortical_subject_path = Path(step1_result["cortical_atlas"])
        subcortical_subject_path = Path(step1_result["subcortical_atlas"])

        # ==================================================
        # STEP 2
        # ==================================================

        logger.info("STEP 2 — Computing ROI Metrics")

        step2_result = compute_roi_metrics(
            jacobian_path=jacobian_path,
            cortical_atlas_path=cortical_subject_path,
            subcortical_atlas_path=subcortical_subject_path,
            roi_config_path=ROI_CONFIG_PATH,
            interval_days=interval_days,
            logger=logger
        )

        with open(output_dir / "step2_roi_metrics.json", "w") as f:
            json.dump(step2_result, f, indent=4)

        # ==================================================
        # STEP 3
        # ==================================================

        logger.info("STEP 3 — Computing Z-Scores")

        step3_result = compute_z_scores(
            roi_step2_output=step2_result,
            age=age,
            normative_path=NORMATIVE_PATH,
            logger=logger
        )

        with open(output_dir / "step3_z_scores.json", "w") as f:
            json.dump(step3_result, f, indent=4)

        # ==================================================
        # STEP 4
        # ==================================================

        logger.info("STEP 4 — Classifying Progression")

        step4_result = classify_progression(
            step3_output=step3_result,
            logger=logger
        )

        with open(output_dir / "step4_classification.json", "w") as f:
            json.dump(step4_result, f, indent=4)

        # ==================================================
        # STEP 5A
        # ==================================================

        logger.info("STEP 5A — Exporting Deterministic T1 Axial Slice")

        t1_png_path = output_dir / "step5_t1_axial_slice.png"

        export_t1_axial_slice(
            t1_nifti_path=t1_followup_path,
            output_png_path=t1_png_path
        )

        # ==================================================
        # STEP 5B
        # ==================================================

        logger.info("STEP 5B — Building Intelligence Payload")

        context_images = {
            "jacobian_overlay": jacobian_overlay_path,
            "t1_followup_slice": t1_png_path
        }

        step5_payload = build_intelligence_payload(
            step2_output=step2_result,
            step3_output=step3_result,
            step4_output=step4_result,
            age=age,
            sex=sex,
            context_images=context_images,
            logger=logger
        )

        with open(output_dir / "step5_payload.json", "w") as f:
            json.dump(step5_payload, f, indent=4)

        # ==================================================
        # STEP 6 — MULTI-STAGE REASONING
        # ==================================================

        logger.info("STEP 6 — Running MedGemma Multi-Stage Reasoning")

        medgemma = get_medgemma_client(logger) 

        class PromptWrapper:
            build_numeric_prompt = staticmethod(build_numeric_prompt)
            build_multimodal_prompt = staticmethod(build_multimodal_prompt)
            build_verification_prompt = staticmethod(build_verification_prompt)
            build_simplification_prompt = staticmethod(build_simplification_prompt)

        step6_result = run_multistage_reasoning(
            payload=step5_payload,
            prompt_builder=PromptWrapper,
            medgemma_client=medgemma.generate,
            output_dir=output_dir
        )

        with open(output_dir / "step6_reasoning.json", "w") as f:
            json.dump(step6_result, f, indent=4)

        # ==================================================
        # FINAL OUTPUT
        # ==================================================

        final_output = {
            "metadata": {
                "age": age,
                "sex": sex,
                "interval_days": interval_days
            },
            "step1": step1_result,
            "step2": step2_result,
            "step3": step3_result,
            "step4": step4_result,
            "step5_payload": step5_payload,
            "step6_reasoning": step6_result
        }

        with open(output_dir / "module2_results.json", "w") as f:
            json.dump(final_output, f, indent=4)

        logger.info("Module 2 completed successfully.")
        logger.info("========================================")

        return final_output

    except Exception as e:
        logger.error("Module 2 failed.")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise

# ==========================================================
# CLI ENTRY
# ==========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Module-2 Quantitative Engine")

    parser.add_argument("--session_dir", type=Path, required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", type=str, required=True)
    parser.add_argument("--interval_days", type=float, required=True)

    args = parser.parse_args()

    session_dir = args.session_dir

    module1_dir = session_dir / "module1"
    module2_dir = session_dir / "module2"

    jacobian_path = module1_dir / "04_ants_syn/jacobian_ants.nii.gz"
    t0_path = module1_dir / "02_bias_corrected/T0/T0_bet_cropped_n4.nii.gz"

    module2_dir.mkdir(parents=True, exist_ok=True)

    result = run_module2(
        jacobian_path=jacobian_path,
        t0_path=t0_path,
        age=args.age,
        sex=args.sex,
        interval_days=args.interval_days,
        output_dir=module2_dir,
        t1_followup_path=module1_dir / "02_bias_corrected/T1/T1_bet_cropped_n4.nii.gz",
        jacobian_overlay_path=module1_dir / "05_visualization/jacobian_overlay.png"
    )

    print("Module-2 completed.")
    print(module2_dir / "module2_results.json")

# if __name__ == "__main__":

#     result = run_module2(
#         jacobian_path=Path("Data/outputs/module1/04_ants_syn/jacobian_ants.nii.gz"),
#         t0_path=Path("Data/outputs/module1/02_bias_corrected/T0/T0_bet_cropped_n4.nii.gz"),
#         age=62,
#         sex="M",
#         interval_days=517,
#         output_dir=Path("Data/outputs/module2"),
#     )

#     print(json.dumps(result, indent=4))
