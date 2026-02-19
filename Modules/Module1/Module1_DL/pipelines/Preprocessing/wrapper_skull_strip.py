"""
Module 1 — Skull Stripping Wrapper

Purpose:
- Provide a stable, CPU-safe skull stripping interface
- Hide HD-BET implementation details
- Allow future backend replacement without touching Module-1 pipeline
"""

from pathlib import Path
import logging

from Modules.Module1.preprocess.hd_bet_pipeline import HDBETPipeline


class SkullStripper:
    """
    Module-1 compliant skull stripping interface.
    """

    def __init__(
        self,
        logger: logging.Logger,
        keep_intermediate: bool = False,
    ):
        self.logger = logger
        self.keep_intermediate = keep_intermediate

        self.logger.info(
            "SkullStripper initialized | Backend=HD-BET | Mode=CPU-only (Blackwell-safe)"
        )

    def run(
        self,
        input_nifti: Path,
        output_dir: Path,
    ) -> Path:
        """
        Execute skull stripping.

        Parameters
        ----------
        input_nifti : Path
            Input MRI (.nii / .nii.gz)
        output_dir : Path
            Output directory

        Returns
        -------
        Path
            Skull-stripped, cropped brain image
        """

        pipeline = HDBETPipeline(
            input_file=input_nifti,
            output_dir=output_dir,
            use_gpu=False,            # 🔒 FORCED CPU
            use_tta=False,            # Deterministic
            keep_intermediate=self.keep_intermediate,
            logger=self.logger,
        )

        pipeline.run()

        self.logger.info("Skull stripping completed successfully")

        return pipeline.final_output
