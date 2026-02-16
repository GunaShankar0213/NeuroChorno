"""
STEP 1 — Register MNI Template to Subject (Fast, Nonlinear)
Then Warp Atlas Labels to Subject Space

Uses antsRegistrationSyNQuick[s] for speed.
Designed for Module 2 ROI extraction.
"""

import ants
import os
import shutil
from pathlib import Path
import logging
import time


# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_THREADS = 16
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(DEFAULT_THREADS)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def register_atlas_to_subject(
    t0_path: Path,
    mni_template_path: Path,
    atlas_cortical_path: Path,
    atlas_subcortical_path: Path,
    output_dir: Path,
    logger: logging.Logger,
):
    """
    Register MNI template to subject T0 space,
    then warp atlas labels into subject space.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading subject (T0)...")
    subject = ants.image_read(str(t0_path))

    logger.info("Loading MNI template...")
    mni = ants.image_read(str(mni_template_path))

    logger.info("Running antsRegistrationSyNQuick[s] (MNI -> Subject)...")

    start_time = time.time()

    registration = ants.registration(
        fixed=subject,
        moving=mni,
        type_of_transform="antsRegistrationSyNQuick[s]"
    )

    runtime_minutes = (time.time() - start_time) / 60.0
    logger.info(f"Registration completed in {runtime_minutes:.2f} minutes")

    # ---------------------------------------------------
    # Save transforms safely (Windows-safe version)
    # ---------------------------------------------------

    logger.info("Saving transforms...")

    fwd_transforms = registration["fwdtransforms"]
    inv_transforms = registration["invtransforms"]

    new_fwd = []
    for i, tf in enumerate(fwd_transforms):

        tf_path = Path(tf)

        if not tf_path.exists():
            raise FileNotFoundError(f"Forward transform not found: {tf}")

        # Preserve full filename extension correctly
        dest = output_dir / f"mni_to_subject_fwd_{i}{tf_path.name[len(tf_path.stem):]}"

        # Better: just use original name extension safely
        if tf_path.name.endswith(".nii.gz"):
            dest = output_dir / f"mni_to_subject_fwd_{i}.nii.gz"
        elif tf_path.suffix == ".mat":
            dest = output_dir / f"mni_to_subject_fwd_{i}.mat"
        else:
            dest = output_dir / f"mni_to_subject_fwd_{i}{tf_path.suffix}"

        shutil.copy2(tf_path, dest)
        logger.info(f"Saved forward transform: {dest.name}")

        new_fwd.append(str(dest))


    new_inv = []
    for i, tf in enumerate(inv_transforms):

        tf_path = Path(tf)

        if not tf_path.exists():
            raise FileNotFoundError(f"Inverse transform not found: {tf}")

        if tf_path.name.endswith(".nii.gz"):
            dest = output_dir / f"subject_to_mni_inv_{i}.nii.gz"
        elif tf_path.suffix == ".mat":
            dest = output_dir / f"subject_to_mni_inv_{i}.mat"
        else:
            dest = output_dir / f"subject_to_mni_inv_{i}{tf_path.suffix}"

        shutil.copy2(tf_path, dest)
        logger.info(f"Saved inverse transform: {dest.name}")

        new_inv.append(str(dest))

    fwd_transforms = new_fwd
    inv_transforms = new_inv

    # -----------------------------
    # Warp Atlas Labels
    # -----------------------------

    logger.info("Warping cortical atlas...")
    atlas_cort = ants.image_read(str(atlas_cortical_path))

    warped_cort = ants.apply_transforms(
        fixed=subject,
        moving=atlas_cort,
        transformlist=fwd_transforms,
        interpolator="nearestNeighbor"
    )

    cort_out = output_dir / "atlas_cortical_in_subject.nii.gz"
    ants.image_write(warped_cort, str(cort_out))

    logger.info("Warping subcortical atlas...")
    atlas_sub = ants.image_read(str(atlas_subcortical_path))

    warped_sub = ants.apply_transforms(
        fixed=subject,
        moving=atlas_sub,
        transformlist=fwd_transforms,
        interpolator="nearestNeighbor"
    )

    sub_out = output_dir / "atlas_subcortical_in_subject.nii.gz"
    ants.image_write(warped_sub, str(sub_out))

    logger.info("Atlas registration complete.")

    return {
        "runtime_minutes": runtime_minutes,
        "threads": DEFAULT_THREADS,
        "cortical_atlas": str(cort_out),
        "subcortical_atlas": str(sub_out),
        "forward_transforms": fwd_transforms,
        "inverse_transforms": inv_transforms
    }


# -----------------------------
# CLI ENTRY
# -----------------------------
if __name__ == "__main__":

    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("MODULE2-ATLAS")

    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", required=True, type=Path, help="T0 image (subject space)")
    parser.add_argument("--mni", required=True, type=Path, help="MNI template")
    parser.add_argument("--atlas-cort", required=True, type=Path, help="Harvard-Oxford cortical atlas")
    parser.add_argument("--atlas-sub", required=True, type=Path, help="Harvard-Oxford subcortical atlas")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")

    args = parser.parse_args()

    result = register_atlas_to_subject(
        t0_path=args.t0,
        mni_template_path=args.mni,
        atlas_cortical_path=args.atlas_cort,
        atlas_subcortical_path=args.atlas_sub,
        output_dir=args.out,
        logger=logger
    )

    print("\nRegistration Summary:")
    print(result)
