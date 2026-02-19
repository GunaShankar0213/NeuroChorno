"""
STEP 3 — Affine Registration (Platinum Standard)
Module: Module 1 — Hybrid Morphometry Engine

PURPOSE:
Performs classical affine registration (T1 -> T0) using SimpleITK.
Automatically generates a Quality Control (QC) Checkerboard PNG.

INPUTS:
- Fixed:  Data/outputs/bias_corrected/T0/result_image_n4.nii.gz
- Moving: Data/outputs/bias_corrected/T1/result_image_n4.nii.gz

OUTPUTS (in Data/outputs/affine/):
- T1_affine_aligned.nii.gz (The aligned image)
- affine_transform.tfm (The math matrix)
- QC_Checkerboard.png (Visual proof of alignment)

Usage:
    python -m pipelines.affine_register_combined
"""

import argparse
import logging
import time
import sys
import os
from pathlib import Path

# Force Headless Mode for Matplotlib (Prevents window errors)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import SimpleITK as sitk

# ------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------
LOGGER_NAME = "STEP3-PLATINUM"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(LOGGER_NAME)

class AffineRegistrationPipeline:
    def __init__(self, fixed_path, moving_path, output_dir):
        self.fixed_path = Path(fixed_path).resolve()
        self.moving_path = Path(moving_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        
        # Create output dir if missing
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output File Paths
        self.warped_moving = self.output_dir / "T1_affine_aligned.nii.gz"
        self.transform_path = self.output_dir / "affine_transform.tfm"
        self.qc_png = self.output_dir / "QC_Checkerboard.png"

    def generate_qc_checkerboard(self, fixed, aligned):
        """Generates a checkerboard overlay of Fixed vs Aligned Moving."""
        logger.info("Generating QC Checkerboard...")
        
        checker = sitk.CheckerBoardImageFilter()
        checker.SetCheckerPattern([2, 2, 1])
        c_img = checker.Execute(fixed, aligned)
        
        # Get Middle Slice
        nda = sitk.GetArrayFromImage(c_img)
        mid_slice = nda.shape[0] // 2
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(nda[mid_slice, :, :], cmap="gray")
        plt.title(f"QC: Checkerboard Alignment\n(Edges should be continuous)", fontsize=14)
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(str(self.qc_png), dpi=150)
        plt.close()
        logger.info(f"   QC Image: {self.qc_png}")

    def run(self):
        logger.info("-" * 40)
        logger.info("STEP 3: AFFINE REGISTRATION & QC")
        logger.info("-" * 40)
        
        if not self.fixed_path.exists():
            logger.error(f"Missing Fixed Image: {self.fixed_path}")
            sys.exit(1)
        if not self.moving_path.exists():
            logger.error(f"Missing Moving Image: {self.moving_path}")
            sys.exit(1)

        logger.info(f"Fixed (T0):  {self.fixed_path.name}")
        logger.info(f"Moving (T1): {self.moving_path.name}")
        logger.info(f"Output Dir:  {self.output_dir}")

        t0 = time.time()

        # 1. Load Images
        fixed = sitk.ReadImage(str(self.fixed_path), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(self.moving_path), sitk.sitkFloat32)

        # 2. Initialize Transform
        logger.info("Initializing Geometry...")
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # 3. Configure Registration
        R = sitk.ImageRegistrationMethod()
        
        # Metric: Mattes Mutual Information
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.01)

        # Optimizer: Gradient Descent
        R.SetOptimizerAsGradientDescent(
            learningRate=1.0, 
            numberOfIterations=100, 
            convergenceMinimumValue=1e-6, 
            convergenceWindowSize=10
        )
        R.SetOptimizerScalesFromPhysicalShift()

        # Multi-Resolution Pyramid
        R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        R.SetInitialTransform(initial_transform, inPlace=False)

        # 4. Execute Registration
        logger.info("Running Optimization (SimpleITK)...")
        final_transform = R.Execute(fixed, moving)

        # 5. Apply Transform (Resample)
        logger.info("Resampling Moving Image...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)

        aligned_image = resampler.Execute(moving)

        # 6. Save Outputs
        sitk.WriteTransform(final_transform, str(self.transform_path))
        sitk.WriteImage(aligned_image, str(self.warped_moving))
        
        elapsed = time.time() - t0
        logger.info(f"✅ Registration Complete ({elapsed:.2f}s)")
        logger.info(f"   Metric Value: {R.GetMetricValue():.4f}")
        logger.info(f"   Saved NIfTI: {self.warped_moving}")
        logger.info(f"   Saved TFM:   {self.transform_path}")

        # 7. Generate QC
        self.generate_qc_checkerboard(fixed, aligned_image)

if __name__ == "__main__":
    # Hardcoded Default Paths (matches your request)
    DEFAULT_FIXED = "Data/outputs/bias_corrected/T0/result_image_n4.nii.gz"
    DEFAULT_MOVING = "Data/outputs/bias_corrected/T1/result_image_n4.nii.gz"
    DEFAULT_OUT = "Data/outputs/affine"

    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", default=DEFAULT_FIXED, help="Fixed Image Path")
    parser.add_argument("--moving", default=DEFAULT_MOVING, help="Moving Image Path")
    parser.add_argument("--output-dir", default=DEFAULT_OUT, help="Output Directory")
    args = parser.parse_args()

    pipeline = AffineRegistrationPipeline(args.fixed, args.moving, args.output_dir)
    pipeline.run()