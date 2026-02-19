"""
Module 1 — Full Hybrid Morphometry Runner (CLI Driven)

Usage
-----
python -m pipelines.module1_full_runner --t0 <T0.nii.gz> --t1 <T1.nii.gz>

Runs entire Module-1 pipeline automatically.
"""

from pathlib import Path
import logging
import sys
import time
import argparse

from pipelines.Preprocessing.wrapper_skull_strip import SkullStripper
from pipelines.Preprocessing.bias_correction import run_bias_correction
from pipelines.Preprocessing.affine_register import AffineRegistrationPipeline

from Modules.Module1.DL_register import run_dl_registration
from Modules.Module1.qa_deformation import qa_check
from Modules.Module1.flow_to_nifti import flow_to_nifti
from Modules.Module1.jacobian import compute_jacobian_determinant
from Modules.Module1.visualize_jacobian import visualize_jacobian_overlay


# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

def configure_logger(log_file: Path):

    logger = logging.getLogger("module1.runner")
    logger.setLevel("INFO")

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------
# Timing wrapper
# ---------------------------------------------------------------------

def timed(logger, name, fn, *args, **kwargs):
    logger.info(f"START: {name}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    dt = time.time() - t0
    logger.info(f"END: {name} | {dt:.2f} sec")
    return result


# ---------------------------------------------------------------------
# Per-timepoint preprocessing
# ---------------------------------------------------------------------

def preprocess_tp(label, input_path, outputs_root, logger):

    skull_dir = outputs_root / "skull_strip" / label
    bias_dir = outputs_root / "bias_corrected" / label

    skull_dir.mkdir(parents=True, exist_ok=True)
    bias_dir.mkdir(parents=True, exist_ok=True)

    skull = SkullStripper(logger=logger, keep_intermediate=False)

    skull_out = timed(
        logger,
        f"{label} SkullStrip",
        skull.run,
        input_path,
        skull_dir
    )

    skull_final = skull_dir / "result_image.nii.gz"
    skull_out.rename(skull_final)

    timed(
        logger,
        f"{label} BiasCorrection",
        run_bias_correction,
        skull_final,
        bias_dir
    )

    bias_file = sorted(bias_dir.glob("*_n4.nii.gz"))[0]
    return bias_file


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline(t0_path: Path, t1_path: Path):

    outputs_root = Path("Data/outputs").resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(outputs_root / "module1_full.log")

    logger.info("========== MODULE-1 START ==========")
    logger.info(f"T0: {t0_path}")
    logger.info(f"T1: {t1_path}")

    if not t0_path.exists() or not t1_path.exists():
        logger.error("Input files not found")
        sys.exit(1)

    total_start = time.time()

    # ---------------------------------------------------------
    # STEP 1–2 — Preprocessing
    # ---------------------------------------------------------

    t0_bias = preprocess_tp("T0", t0_path, outputs_root, logger)
    t1_bias = preprocess_tp("T1", t1_path, outputs_root, logger)

    # ---------------------------------------------------------
    # STEP 3 — Affine registration
    # ---------------------------------------------------------

    affine_dir = outputs_root / "affine"
    affine_dir.mkdir(exist_ok=True)

    affine = AffineRegistrationPipeline(
        fixed_path=t0_bias,
        moving_path=t1_bias,
        output_dir=affine_dir
    )

    timed(logger, "AffineRegistration", affine.run)

    moving_affine = affine_dir / "T1_affine_aligned.nii.gz"

    # ---------------------------------------------------------
    # STEP 4 — DL registration
    # ---------------------------------------------------------

    dl_dir = outputs_root / "dl_registration"
    dl_dir.mkdir(exist_ok=True)

    warped, flow = timed(
        logger,
        "DLRegistration",
        run_dl_registration,
        fixed_path=t0_bias,
        moving_path=moving_affine,
        model_path=r"model_store\TransMorph-diff\transmorph_large\TransMorphLarge.pth.tar",
        out_dir=dl_dir
    )

    # IMPORTANT: DL module writes resampled fixed image here
    fixed_rs = dl_dir / "fixed_rs.nii.gz"

    if not fixed_rs.exists():
        logger.error("DL resampled fixed image missing — cannot run QA safely")
        sys.exit(3)

    # ---------------------------------------------------------
    # STEP 5 — QA gate (FIXED — uses fixed_rs)
    # ---------------------------------------------------------

    qa_json = outputs_root / "qa_metrics.json"

    accepted, metrics = timed(
        logger,
        "DeformationQA",
        qa_check,
        flow,
        fixed_rs,   # ✅ FIX — same grid as warped
        warped,
        qa_json
    )

    logger.info(metrics)

    if not accepted:
        logger.error("QA failed — stopping pipeline")
        sys.exit(2)

    # ---------------------------------------------------------
    # STEP 6 — Flow → warp nifti
    # ---------------------------------------------------------

    warp_nifti = timed(
        logger,
        "FlowToNifti",
        flow_to_nifti,
        flow,
        fixed_rs,   # keep same grid
        dl_dir / "warp_field_dl.nii.gz"
    )

    # ---------------------------------------------------------
    # STEP 7 — Jacobian map
    # ---------------------------------------------------------

    jac_path = timed(
        logger,
        "Jacobian",
        compute_jacobian_determinant,
        warp_nifti,
        outputs_root / "jacobian_map.nii.gz",
        True
    )

    # ---------------------------------------------------------
    # STEP 8 — Visualization
    # ---------------------------------------------------------

    timed(
        logger,
        "Visualization",
        visualize_jacobian_overlay,
        jac_path,
        fixed_rs,
        outputs_root / "jacobian_overlay.png"
    )

    total = time.time() - total_start
    logger.info(f"TOTAL RUNTIME: {total/60:.2f} minutes")
    logger.info("========== MODULE-1 COMPLETE ==========")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True)
    ap.add_argument("--t1", required=True)

    args = ap.parse_args()

    run_pipeline(Path(args.t0), Path(args.t1))


if __name__ == "__main__":
    main()


### missed the fall-backs ANT's 

"""
negative Jacobians %
You compute Jacobian after QA.

Better future improvement:

add negative-Jacobian check after jacobian step

Deviation 1 — Affine Tool

Spec:

ANTs affine


You used:

SimpleITK affine


Scientifically equivalent — safe.
"""