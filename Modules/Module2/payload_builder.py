"""
STEP 5 — Intelligence Payload Builder

Context images are mandatory.
No model inference occurs here.
Pure deterministic data preparation.
"""

from pathlib import Path
import logging
from typing import Dict


REQUIRED_ROIS = [
    "hippocampus",
    "entorhinal_cortex",
    "temporal_lobe",
    "parietal_lobe",
    "ventricles"
]


def _validate_step2(step2: Dict):
    if "roi_metrics" not in step2:
        raise ValueError("Step 2 output missing 'roi_metrics'.")
    if "interval_years" not in step2:
        raise ValueError("Step 2 output missing 'interval_years'.")


def _validate_step3(step3: Dict):
    if "z_scores" not in step3:
        raise ValueError("Step 3 output missing 'z_scores'.")
    for roi in REQUIRED_ROIS:
        if roi not in step3["z_scores"]:
            raise ValueError(f"Missing ROI in Step 3: {roi}")
        if "z_score" not in step3["z_scores"][roi]:
            raise ValueError(f"Missing z_score for ROI: {roi}")
        if "annual_percent_change" not in step3["z_scores"][roi]:
            raise ValueError(f"Missing annual_percent_change for ROI: {roi}")


def _validate_step4(step4: Dict):
    if "progression_class" not in step4:
        raise ValueError("Step 4 output missing 'progression_class'.")
    if "score" not in step4:
        raise ValueError("Step 4 output missing 'score'.")
    if "rationale" not in step4:
        raise ValueError("Step 4 output missing 'rationale'.")


def _validate_context_images(context_images: Dict[str, Path]):
    required_keys = ["jacobian_overlay", "t1_followup_slice"]

    for key in required_keys:
        if key not in context_images:
            raise ValueError(f"Missing required context image: {key}")

        if not Path(context_images[key]).exists():
            raise FileNotFoundError(
                f"Context image not found: {context_images[key]}"
            )


def build_intelligence_payload(
    step2_output: Dict,
    step3_output: Dict,
    step4_output: Dict,
    age: int,
    sex: str,
    context_images: Dict[str, Path],
    logger: logging.Logger
) -> Dict:
    """
    Build sanitized intelligence payload for MedGemma.
    Context images are REQUIRED.
    """

    logger.info("Validating Step 2 output...")
    _validate_step2(step2_output)

    logger.info("Validating Step 3 output...")
    _validate_step3(step3_output)

    logger.info("Validating Step 4 output...")
    _validate_step4(step4_output)

    logger.info("Validating context images...")
    _validate_context_images(context_images)

    interval_years = round(step2_output["interval_years"], 3)

    roi_block = {}

    for roi in REQUIRED_ROIS:
        annual = step3_output["z_scores"][roi]["annual_percent_change"]
        z_val = step3_output["z_scores"][roi]["z_score"]

        roi_block[roi] = {
            "annual_percent_change": round(float(annual), 3),
            "z_score": round(float(z_val), 3)
        }

    payload = {
        "metadata": {
            "age": int(age),
            "sex": str(sex),
            "interval_years": interval_years
        },
        "roi_metrics": roi_block,
        "progression": {
            "class": step4_output["progression_class"],
            "score": int(step4_output["score"]),
            "rationale": step4_output["rationale"]
        },
        "context_images": {
            "jacobian_overlay": str(context_images["jacobian_overlay"]),
            "t1_followup_slice": str(context_images["t1_followup_slice"])
        }
    }

    logger.info("Step 5 intelligence payload constructed successfully.")

    return payload
