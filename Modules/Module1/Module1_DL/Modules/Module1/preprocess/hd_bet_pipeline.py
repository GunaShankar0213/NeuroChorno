"""
Docstring for Modules.Module1.preprocess.skull_strip
"""


import time
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

LOGGER_NAME = "hd_bet_pipeline"


def configure_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    logger.setLevel(level.upper())

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------
# Pipeline Implementation
# ---------------------------------------------------------------------

class HDBETPipeline:
    """
    Production-ready HD-BET skull stripping pipeline.

    Responsibilities:
    - Execute HD-BET with optional GPU and TTA
    - Perform affine-safe auto-cropping
    - Generate QC visualization
    - Optional cleanup of intermediate artifacts

    Designed for reuse as both:
    - CLI tool
    - Importable Python module
    """

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        use_gpu: bool = False,
        use_tta: bool = False,
        keep_intermediate: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(LOGGER_NAME)

        self.input_file = input_file
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.use_tta = use_tta
        self.keep_intermediate = keep_intermediate

        self.logger.debug("Initializing HDBETPipeline")
        self.logger.debug(f"Input file: {self.input_file}")
        self.logger.debug(f"Output directory: {self.output_dir}")
        self.logger.debug(f"GPU requested: {self.use_gpu}")
        self.logger.debug(f"TTA requested: {self.use_tta}")
        self.logger.debug(f"Keep intermediate files: {self.keep_intermediate}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        stem = input_file.stem.replace(".nii", "")

        # Intermediate files
        self.bet_output = self.output_dir / f"{stem}_bet.nii.gz"
        self.bet_mask_output = self.output_dir / f"{stem}_bet_mask.nii.gz"

        # Final deliverables
        self.final_output = self.output_dir / f"{stem}_bet_cropped.nii.gz"
        self.qc_report = self.output_dir / f"{stem}_qc_report.png"

    @staticmethod
    def _check_gpu_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def _crop_with_affine_correction(
        img_obj: nib.Nifti1Image, # type: ignore
        logger: logging.Logger,
    ) -> nib.Nifti1Image: # type: ignore
        logger.info("Starting affine-safe auto-cropping")

        data = img_obj.get_fdata()
        affine = img_obj.affine

        non_zero = np.argwhere(data > 0)
        if non_zero.size == 0:
            raise ValueError("Input image is empty after skull stripping")

        x_min, y_min, z_min = non_zero.min(axis=0)
        x_max, y_max, z_max = non_zero.max(axis=0) + 1

        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        z_min = max(0, z_min - margin)
        x_max = min(data.shape[0], x_max + margin)
        y_max = min(data.shape[1], y_max + margin)
        z_max = min(data.shape[2], z_max + margin)

        cropped_data = data[x_min:x_max, y_min:y_max, z_min:z_max]

        origin_shift = np.array([x_min, y_min, z_min, 1])
        new_origin = affine @ origin_shift

        new_affine = affine.copy() # type: ignore
        new_affine[:3, 3] = new_origin[:3]

        new_header = img_obj.header.copy()
        new_header.set_data_shape(cropped_data.shape)

        logger.info(f"Cropping complete | New shape: {cropped_data.shape}")

        return nib.Nifti1Image(cropped_data, new_affine, new_header) # type: ignore
 
    @staticmethod
    def _save_qc_report(
        original_path: Path,
        final_img: nib.Nifti1Image, # type: ignore
        output_path: Path,
        logger: logging.Logger,
    ) -> None:
        logger.info("Generating QC visualization")

        orig_img = nib.load(original_path) # type: ignore
        orig_data = orig_img.get_fdata() # type: ignore
        final_data = final_img.get_fdata()

        mid_orig = orig_data.shape[0] // 2
        mid_final = final_data.shape[0] // 2

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(np.rot90(orig_data[mid_orig, :, :]), cmap="gray")
        axes[0].set_title("Original Input")
        axes[0].axis("off")

        axes[1].imshow(np.rot90(final_data[mid_final, :, :]), cmap="gray")
        axes[1].set_title("Skull Stripped & Cropped")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info(f"QC report saved: {output_path}")

    def run(self) -> None:
        self.logger.info("Starting HD-BET pipeline")

        if not self.input_file.exists():
            self.logger.error(f"Input file not found: {self.input_file}")
            raise FileNotFoundError(self.input_file)

        device_flag = "cpu"
        if self.use_gpu:
            if self._check_gpu_available():
                device_flag = "cuda:0"
                self.logger.info("GPU detected and enabled")
            else:
                self.logger.warning("GPU requested but not available; using CPU")

        cmd = [
            "hd-bet",
            "-i", str(self.input_file),
            "-o", str(self.bet_output),
            "-device", device_flag,
            "--save_bet_mask",
        ]

        if not self.use_tta:
            cmd.append("--disable_tta")

        self.logger.debug(f"HD-BET command: {' '.join(cmd)}")

        try:
            self.logger.info("Running HD-BET inference")
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.logger.info("HD-BET completed successfully")
            time.sleep(3)

            bet_img = nib.load(self.bet_output) # type: ignore

            cropped_img = self._crop_with_affine_correction(
                bet_img, # type: ignore
                self.logger,
            )

            nib.save(cropped_img, self.final_output) # type: ignore
            self.logger.info(f"Final NIfTI saved: {self.final_output}")

            self._save_qc_report(
                self.input_file,
                cropped_img,
                self.qc_report,
                self.logger,
            )

        finally:
            if self.keep_intermediate:
                self.logger.info("Keeping intermediate files as requested")
                return

            self.logger.info("Cleaning intermediate files")

            if self.bet_output.exists():
                self.bet_output.unlink()
                self.logger.debug(f"Removed: {self.bet_output}")

            if self.bet_mask_output.exists():
                self.bet_mask_output.unlink()
                self.logger.debug(f"Removed: {self.bet_mask_output}")

            self.logger.info("Intermediate cleanup complete")


# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HD-BET Skull Stripping Pipeline")

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input NIfTI file (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "processed",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU execution if available",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Test-Time Augmentation",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Retain intermediate HD-BET outputs (_bet, _bet_mask)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    args = parser.parse_args()

    logger = configure_logger(args.log_level)

    pipeline = HDBETPipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        use_gpu=args.gpu,
        use_tta=args.tta,
        keep_intermediate=args.keep_intermediate,
        logger=logger,
    )

    pipeline.run()


if __name__ == "__main__":
    main()


"""
Default (clean workspace)
python hd_bet_pipeline.py \
  --input T0.nii.gz \
  --output-dir processed

Debugging / QA / Research Mode
python hd_bet_pipeline.py \
  --input T0.nii.gz \
  --output-dir processed \
  --keep-intermediate \
  --log-level DEBUG

  GPU enabled (safe default)
python hd_bet_pipeline.py \
  --input T0.nii.gz \
  --gpu

GPU + High Quality (recommended use case)
python hd_bet_pipeline.py \
  --input T0.nii.gz \
  --gpu \
  --tta

Explicit CPU-only (for reproducibility)
python hd_bet_pipeline.py \
  --input T0.nii.gz
"""