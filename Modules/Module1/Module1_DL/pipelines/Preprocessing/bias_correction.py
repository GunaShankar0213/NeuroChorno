"""
STEP 2 — Bias Field Correction (Platinum Standard - Enhanced)
Module: Morphometry Pre-processing

ENHANCEMENTS:
- HEADLESS VISUALS: Uses 'Agg' backend to guarantee PNG saving.
- GEOMETRY FIX: Correctly handles NIfTI (x,y,z) vs SimpleITK (z,y,x).
- SAFETY WRITE: double-checks directory existence before saving.
- THREADING: Hardcoded to 20 threads.
- PROGRESS: Tqdm visualization.

Usage:
    python -m pipelines.bias_correction --input <file> --output-dir <dir>
"""

import argparse
import logging
import time
import multiprocessing
import sys
import os
from pathlib import Path

# FIX: Force Headless Mode
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("STEP2-PLATINUM")

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def nib_to_sitk(nib_img):
    """Safely converts Nibabel NIfTI to SimpleITK Image."""
    data_xyz = nib_img.get_fdata(dtype=np.float32)
    data_zyx = data_xyz.T 
    img_sitk = sitk.GetImageFromArray(data_zyx)
    
    header = nib_img.header
    spacing_xyz = header.get_zooms()[:3]
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    
    img_sitk.SetSpacing(spacing_zyx)
    return img_sitk

def sitk_to_nib(sitk_img, reference_nib):
    """Converts SimpleITK Image back to Nibabel."""
    data_zyx = sitk.GetArrayFromImage(sitk_img)
    data_xyz = data_zyx.T
    return nib.Nifti1Image(data_xyz, reference_nib.affine, reference_nib.header) #type: ignore

def save_visual_report(original_nib, corrected_nib, out_path):
    """Generates a slice view to verify contrast improvement."""
    try:
        orig = original_nib.get_fdata()
        corr = corrected_nib.get_fdata()
        mid_slice = orig.shape[0] // 2 
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        vmin = np.percentile(orig, 1)
        vmax = np.percentile(orig, 99)

        axes[0].imshow(np.rot90(orig[mid_slice, :, :]), cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("Before N4 (Raw)")
        axes[0].axis("off")
        
        axes[1].imshow(np.rot90(corr[mid_slice, :, :]), cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title("After N4 (Corrected)")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close(fig)
        return True
    except Exception as e:
        logger.error(f"Failed to generate PNG: {e}")
        return False

# ------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------
def run_bias_correction(input_path: Path, output_dir: Path):
    t0 = time.time()
    
    # 1. Hardware Optimization
    target_threads = 20
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(target_threads)
    
    # Fix: Ensure absolute paths
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    
    # Create directory immediately
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    out_nifti = output_dir / f"{base_name}_n4.nii.gz"
    out_png = output_dir / f"{base_name}_n4.png"
    
    logger.info(f"Processing: {input_path.name}")
    
    # Progress Bar Context
    with tqdm(total=5, desc="N4 Pipeline", unit="step") as pbar:
        
        # STEP 1: Load Data
        pbar.set_description("Loading NIfTI")
        nib_img = nib.load(str(input_path))  #type: ignore
        sitk_img = nib_to_sitk(nib_img)
        pbar.update(1)
        
        # STEP 2: Intelligent Masking
        pbar.set_description("Generating Mask")
        mask_img = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        
        spacing = sitk_img.GetSpacing()
        radius_voxels = [max(1, int(round(1.0 / s))) for s in spacing]
        
        dilater = sitk.BinaryDilateImageFilter()
        dilater.SetKernelRadius(radius_voxels)
        mask_img = dilater.Execute(mask_img)
        pbar.update(1)
        
        # STEP 3: Configure N4
        pbar.set_description("Configuring N4")
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50]) 
        corrector.SetConvergenceThreshold(0.001)
        corrector.SetNumberOfWorkUnits(target_threads)
        pbar.update(1)
        
        # STEP 4: Execute
        pbar.set_description("Running Optimization")
        corrected_sitk = corrector.Execute(sitk_img, mask_img)
        pbar.update(1)
        
        # STEP 5: Save Results
        pbar.set_description("Saving Output")
        corrected_nib = sitk_to_nib(corrected_sitk, nib_img)
        
        # SAFETY CHECK: Verify directory exists right before saving
        if not out_nifti.parent.exists():
            logger.warning(f"Re-creating missing directory: {out_nifti.parent}")
            out_nifti.parent.mkdir(parents=True, exist_ok=True)
        
        # SAFETY WAIT: Give filesystem 1s to settle
        time.sleep(1.0)
        
        # Force string conversion for Windows safety
        nib.save(corrected_nib, os.path.abspath(str(out_nifti)))  #type: ignore
        save_visual_report(nib_img, corrected_nib, out_png)
        pbar.update(1)

    elapsed = time.time() - t0
    logger.info(f"✅ Success. Time: {elapsed:.2f}s")
    logger.info(f"   NIfTI: {out_nifti}")
    logger.info(f"   PNG:   {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input skull-stripped NIfTI")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    
    # Ensure inputs exist before starting
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    run_bias_correction(args.input, args.output_dir)

"""
python -m pipelines.bias_correction --input Data\outputs\skull_strip\T0\result_image.nii.gz  
--output-dir Data\outputs\bias_corrected\T0

"""