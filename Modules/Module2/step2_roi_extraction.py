"""
STEP 2 — ROI Extraction & Annualized Jacobian Metrics
Dual-atlas aware, QA-hardened implementation
"""

import nibabel as nib
import numpy as np
import json
from pathlib import Path
import logging


def compute_roi_metrics(
    jacobian_path: Path,
    cortical_atlas_path: Path,
    subcortical_atlas_path: Path,
    roi_config_path: Path,
    interval_days: float,
    logger: logging.Logger
):

    # -------------------------------------------------
    # Load Data
    # -------------------------------------------------

    logger.info("Loading Jacobian map...")
    jac = nib.load(str(jacobian_path)).get_fdata()

    logger.info("Loading warped cortical atlas...")
    cortical = nib.load(str(cortical_atlas_path)).get_fdata()

    logger.info("Loading warped subcortical atlas...")
    subcortical = nib.load(str(subcortical_atlas_path)).get_fdata()

    # -------------------------------------------------
    # Integrity Checks
    # -------------------------------------------------

    if jac.shape != cortical.shape:
        raise ValueError("Jacobian and cortical atlas dimensions do not match.")

    if jac.shape != subcortical.shape:
        raise ValueError("Jacobian and subcortical atlas dimensions do not match.")

# -------------------------------------------------
    # Integrity Checks
    # -------------------------------------------------

    if jac.shape != cortical.shape:
        raise ValueError("Jacobian and cortical atlas dimensions do not match.")

    if jac.shape != subcortical.shape:
        raise ValueError("Jacobian and subcortical atlas dimensions do not match.")

    # FIX: Enforce minimum clinical interval to prevent ZeroDivisionError 
    # and noise amplification (e.g., annualizing a 1-day change).
    MIN_INTERVAL_DAYS = 30
    if interval_days < MIN_INTERVAL_DAYS:
        raise ValueError(
            f"Scan interval ({interval_days:.1f} days) is too short for reliable atrophy calculation. "
            f"Minimum {MIN_INTERVAL_DAYS} days required."
        )

    # Enforce diffeomorphic deformation
    if np.any(jac <= 0):
        raise ValueError("Non-diffeomorphic Jacobian detected. Upstream failure.")

    # -------------------------------------------------
    # Load ROI Configuration
    # -------------------------------------------------

    logger.info("Loading ROI configuration...")
    with open(roi_config_path) as f:
        roi_labels = json.load(f)

    # -------------------------------------------------
    # Atlas-Aware ROI Exclusivity Validation
    # -------------------------------------------------

    logger.info("Validating ROI label exclusivity within each atlas...")

    cortical_labels_all = []
    subcortical_labels_all = []

    for roi_name, labels in roi_labels.items():
        if roi_name in ["hippocampus", "ventricles"]:
            subcortical_labels_all.extend(labels)
        else:
            cortical_labels_all.extend(labels)

    if len(cortical_labels_all) != len(set(cortical_labels_all)):
        raise ValueError("Overlapping cortical ROI labels detected.")

    if len(subcortical_labels_all) != len(set(subcortical_labels_all)):
        raise ValueError("Overlapping subcortical ROI labels detected.")

    # -------------------------------------------------
    # ROI Computation
    # -------------------------------------------------

    interval_years = interval_days / 365.25
    results = {}

    required_rois = [
        "hippocampus",
        "entorhinal_cortex",
        "temporal_lobe",
        "parietal_lobe",
        "ventricles"
    ]

    for roi_name, labels in roi_labels.items():

        # Route ROI to correct atlas
        if roi_name in ["hippocampus", "ventricles"]:
            atlas_data = subcortical
        else:
            atlas_data = cortical

        unique_labels = np.unique(atlas_data)

        # Validate labels exist in atlas
        for label in labels:
            if label not in unique_labels:
                raise ValueError(
                    f"Label {label} for ROI '{roi_name}' not found in corresponding atlas."
                )

        mask = np.isin(atlas_data, labels)
        voxel_count = int(mask.sum())

        if roi_name in required_rois and voxel_count == 0:
            raise ValueError(f"Required ROI '{roi_name}' has empty mask.")

        if voxel_count == 0:
            logger.warning(f"{roi_name} mask empty.")
            continue

        values = jac[mask]

        mean_jac = float(np.mean(values))
        percent_change_total = (mean_jac - 1.0) * 100.0
        percent_change_per_year = percent_change_total / interval_years

        results[roi_name] = {
            "mean_jacobian": mean_jac,
            "percent_change_total": percent_change_total,
            "percent_change_per_year": percent_change_per_year,
            "voxel_count": voxel_count
        }

        logger.info(
            f"{roi_name}: {percent_change_per_year:.3f}%/year | Voxels={voxel_count}"
        )

    # -------------------------------------------------
    # Final Output
    # -------------------------------------------------

    return {
        "interval_years": interval_years,
        "roi_metrics": results
    }
