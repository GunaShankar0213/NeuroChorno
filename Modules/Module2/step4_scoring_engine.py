"""
STEP 4 — Fast Progressor Classification (5 ROI Version)
Depends on:
- STEP 3 Z-score output structure
"""

import logging


def classify_progression(step3_output: dict, logger: logging.Logger):

    # -----------------------------
    # Structure Validation
    # -----------------------------

    if "z_scores" not in step3_output:
        raise ValueError("Invalid Step 3 output structure: missing 'z_scores'.")

    if "interval_years" not in step3_output:
        raise ValueError("Step 3 output missing 'interval_years'.")

    z_scores = step3_output["z_scores"]

    required_rois = [
        "hippocampus",
        "entorhinal_cortex",
        "temporal_lobe",
        "parietal_lobe",
        "ventricles"
    ]

    for roi in required_rois:
        if roi not in z_scores:
            raise ValueError(f"Missing ROI in Z-score output: {roi}")

        if "z_score" not in z_scores[roi]:
            raise ValueError(f"Missing z_score for ROI: {roi}")

        if "annual_percent_change" not in z_scores[roi]:
            raise ValueError(f"Missing annual_percent_change for ROI: {roi}")

    # -----------------------------
    # Extract Values
    # -----------------------------

    hip_z = float(z_scores["hippocampus"]["z_score"])
    ent_z = float(z_scores["entorhinal_cortex"]["z_score"])
    temp_z = float(z_scores["temporal_lobe"]["z_score"])
    par_z = float(z_scores["parietal_lobe"]["z_score"])
    vent_z = float(z_scores["ventricles"]["z_score"])

    hip_rate = float(z_scores["hippocampus"]["annual_percent_change"])

    score = 0
    rationale = []

    # -----------------------------
    # Core Alzheimer’s Pattern Rules
    # -----------------------------

    if hip_z < -2.0:
        score += 2
        rationale.append("Hippocampal atrophy Z < -2.0")

    if ent_z < -1.5:
        score += 1
        rationale.append("Entorhinal atrophy Z < -1.5")

    if temp_z < -1.5:
        score += 1
        rationale.append("Temporal lobe atrophy Z < -1.5")

    if par_z < -1.5:
        score += 1
        rationale.append("Parietal lobe atrophy Z < -1.5")

    if vent_z > 2.0:
        score += 1
        rationale.append("Ventricular expansion Z > +2.0")

    # -----------------------------
    # Aggressive Atrophy Rate Rule
    # (Negative direction = tissue loss)
    # -----------------------------

    if hip_rate < -3.5:
        score += 2
        rationale.append("Hippocampal annual atrophy < -3.5%")

    # -----------------------------
    # Final Classification
    # -----------------------------

    classification = (
        "Fast Progressor"
        if score >= 4
        else "Normal Progression"
    )

    logger.info(f"Final classification: {classification} | Score={score}")

    return {
        "progression_class": classification,
        "score": score,
        "rationale": rationale
    }
