"""
STEP 3 — Age-Aware Z-Score Computation
Depends on:
- STEP 2 ROI metrics output structure
"""

import json
from pathlib import Path
import logging


def select_age_bin(age: int, normative_data: dict):
    # Parse all bins into (low, high, key) tuples
    bins = []
    for key in normative_data.keys():
        try:
            low, high = map(int, key.split("-"))
            bins.append((low, high, key))
        except ValueError:
            continue  # Skip malformed keys

    # Sort bins by starting age
    bins.sort(key=lambda x: x[0])

    if not bins:
        raise ValueError("Normative data is empty or invalid.")

    # 1. Try to find the exact correct bin
    for low, high, key in bins:
        if low <= age < high:
            return key, normative_data[key]

    # 2. Clamp Logic (Nearest Neighbor)
    
    # If patient is younger than the youngest bin, use the first bin
    if age < bins[0][0]:
        target_key = bins[0][2]
        return target_key, normative_data[target_key]

    # If patient is older than the oldest bin, use the last bin
    # (The loop above failed, so age must be >= highest bin's end)
    target_key = bins[-1][2]
    return target_key, normative_data[target_key]


def compute_z_scores(
    roi_step2_output: dict,
    age: int,
    normative_path: Path,
    logger: logging.Logger
):

    logger.info("Loading normative reference data...")

    with open(normative_path) as f:
        full_norm = json.load(f)

    # -----------------------------
    # Normative Structure Validation
    # -----------------------------

    if "age_bins" not in full_norm:
        raise ValueError("Invalid normative file: missing 'age_bins'.")

    normative_data = full_norm["age_bins"]

    age_bin_label, age_norm = select_age_bin(age, normative_data)

    # -----------------------------
    # Step 2 Structure Validation
    # -----------------------------

    if "roi_metrics" not in roi_step2_output:
        raise ValueError("Invalid Step 2 output structure.")

    if "interval_years" not in roi_step2_output:
        raise ValueError("Step 2 output missing interval_years.")

    roi_metrics = roi_step2_output["roi_metrics"]
    interval_years = roi_step2_output["interval_years"]

    if interval_years <= 0:
        raise ValueError("Invalid interval_years from Step 2.")

    # -----------------------------
    # Required ROI Enforcement
    # -----------------------------

    required_rois = [
        "hippocampus",
        "entorhinal_cortex",
        "temporal_lobe",
        "parietal_lobe",
        "ventricles"
    ]

    for roi in required_rois:
        if roi not in roi_metrics:
            raise ValueError(f"Missing ROI in Step 2 output: {roi}")

        if roi not in age_norm:
            raise ValueError(f"Missing ROI '{roi}' in normative reference.")

    # -----------------------------
    # Z-Score Computation
    # -----------------------------

    z_results = {}

    for roi in required_rois:

        values = roi_metrics[roi]

        annual = values["percent_change_per_year"]
        mean = age_norm[roi]["mean"]
        std = age_norm[roi]["std"]

        if std == 0:
            raise ValueError(f"Standard deviation for {roi} is zero.")

        z = (annual - mean) / std

        z_results[roi] = {
            "annual_percent_change": annual,
            "normative_mean": mean,
            "normative_std": std,
            "z_score": float(z)
        }

        logger.info(f"{roi}: Z={z:.3f}")

    return {
        "age": age,
        "age_bin": age_bin_label,
        "interval_years": interval_years,
        "z_scores": z_results
    }
