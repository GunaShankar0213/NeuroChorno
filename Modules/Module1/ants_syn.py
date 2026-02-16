"""
ANTs SyN Deformable Registration
Supports both str and pathlib.Path inputs.
Includes:
- Jacobian metrics
- NCC similarity
- Max displacement
- Writes full QC JSON
"""

import ants
import numpy as np
import nibabel as nib
import json
import os
import shutil
from pathlib import Path
from typing import Union
from datetime import datetime
import os

DEFAULT_ANTS_THREADS = 18  # Set according to your CPU
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(DEFAULT_ANTS_THREADS)


PathLike = Union[str, Path]


def _to_str(path: PathLike) -> str:
    """Convert PathLike to string."""
    return str(path)


def compute_ncc(fixed_np: np.ndarray, warped_np: np.ndarray) -> float:
    """Compute masked normalized cross-correlation."""
    mask = (fixed_np > 0) & (warped_np > 0)
    f = fixed_np[mask]
    w = warped_np[mask]
    if f.size == 0:
        raise ValueError("Empty mask in NCC.")
    f = (f - f.mean()) / (f.std() + 1e-8)
    w = (w - w.mean()) / (w.std() + 1e-8)
    return float(np.mean(f * w))


def compute_max_displacement(warp_path: PathLike) -> float:
    """Compute maximum displacement magnitude."""
    warp_img = nib.load(_to_str(warp_path))
    warp = warp_img.get_fdata()

    if warp.shape[-1] != 3:
        raise ValueError("Warp must contain 3 components.")

    disp_mag = np.sqrt(np.sum(np.square(warp), axis=-1))
    return float(np.max(disp_mag))


def run_ants_syn(
    fixed_path: PathLike,
    moving_path: PathLike,
    output_dir: PathLike
):
    """
    Run ANTs SyN deformable registration.

    Parameters
    ----------
    fixed_path : str | Path
    moving_path : str | Path
    output_dir : str | Path
    """

    fixed_path = _to_str(fixed_path)
    moving_path = _to_str(moving_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)

    print("Running ANTs SyN registration...")
    start_time = datetime.utcnow()

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=(100, 70, 50, 20)
    )

    warped = reg["warpedmovout"]
    warp_field = reg["fwdtransforms"][0]
    inverse_warp = reg["invtransforms"][0]

    warped_path = output_dir / "warped_ants.nii.gz"
    warp_path = output_dir / "warp_ants.nii.gz"
    inv_warp_path = output_dir / "inverse_warp_ants.nii.gz"
    jacobian_path = output_dir / "jacobian_ants.nii.gz"
    metrics_path = output_dir / "ants_jacobian_metrics.json"

    ants.image_write(warped, str(warped_path))

    shutil.copy(warp_field, warp_path)
    shutil.copy(inverse_warp, inv_warp_path)

    jacobian = ants.create_jacobian_determinant_image(
        domain_image=fixed,
        tx=warp_field,
        do_log=False
    )
    ants.image_write(jacobian, str(jacobian_path))

    jac_np = jacobian.numpy()

    mean_jac = float(np.mean(jac_np))
    std_jac = float(np.std(jac_np))
    neg_jac_pct = float(np.sum(jac_np <= 0) / jac_np.size * 100)

    fixed_np = fixed.numpy()
    warped_np = warped.numpy()
    ncc_value = compute_ncc(fixed_np, warped_np)

    max_disp = compute_max_displacement(warp_path)

    qa_pass = (
        neg_jac_pct <= 0.2 and
        0.98 <= mean_jac <= 1.02 and
        ncc_value >= 0.90 and
        max_disp <= 15.0
    )

    end_time = datetime.utcnow()
    runtime_minutes = (end_time - start_time).total_seconds() / 60.0

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "mean_jacobian": mean_jac,
        "std_jacobian": std_jac,
        "negative_jacobian_pct": neg_jac_pct,
        "ncc": ncc_value,
        "max_displacement_mm": max_disp,
        "qa_pass": qa_pass,
        "runtime_minutes": runtime_minutes,
        "transform_type": "ANTs_SyN",
        "threads": DEFAULT_ANTS_THREADS,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("ANTs SyN completed.")
    print("QA PASS:", qa_pass)

    return {
        "warped": str(warped_path),
        "warp": str(warp_path),
        "inverse_warp": str(inv_warp_path),
        "jacobian": str(jacobian_path),
        "metrics": str(metrics_path),
        "qa_pass": qa_pass
    }
