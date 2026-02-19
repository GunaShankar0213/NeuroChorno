"""
prepare_simon_engineering_tests.py

Creates Tier A and Tier B engineering validation datasets
from the SIMON BIDS MRI dataset.

Tier A: Short-interval stability test (same subject, minimal change)
Tier B: Long-interval drift test (same subject, years apart)

This script copies (not moves) T1w images to clean,
Module-1-ready folders.

Designed for:
- Windows
- Python 3.12
- Strict typing (Pylance clean)
"""

from pathlib import Path
import shutil


# ============================
# USER CONFIGURATION (EDIT ONLY THIS SECTION)
# ============================

SIMON_BIDS_ROOT = Path(
    "Data/SIMON/SIMON_BIDS/sub-032633"
)

# ---- Tier A: short-interval (baseline stability) ----
TIER_A_SESSION_T0 = "ses-001"
TIER_A_SESSION_T1 = "ses-002"

# ---- Tier B: long-interval (aging drift, still healthy) ----
TIER_B_SESSION_T0 = "ses-001"
TIER_B_SESSION_T1 = "ses-060"

# Output root for Module 1
OUTPUT_ROOT = Path("data/raw")


# ============================
# INTERNAL LOGIC (DO NOT EDIT)
# ============================

def find_t1w(session_dir: Path) -> Path:
    """
    Locate the T1w NIfTI file inside a BIDS session/anat directory.

    Parameters
    ----------
    session_dir : Path
        Path to a session directory (e.g., ses-001)

    Returns
    -------
    Path
        Path to the T1w NIfTI file

    Raises
    ------
    FileNotFoundError
        If no T1w file is found
    """

    anat_dir = session_dir / "anat"

    if not anat_dir.exists():
        raise FileNotFoundError(f"Missing anat directory: {anat_dir}")

    t1w_files = sorted(anat_dir.glob("*_T1w.nii.gz"))

    if not t1w_files:
        raise FileNotFoundError(f"No T1w files found in {anat_dir}")

    # Use the first run deterministically
    return t1w_files[0]


def create_test_pair(
    tier_name: str,
    session_t0: str,
    session_t1: str
) -> None:
    """
    Create a Module-1-ready test pair for a given tier.

    Parameters
    ----------
    tier_name : str
        Name of the tier (e.g., "simon_tier_A")
    session_t0 : str
        Baseline session ID
    session_t1 : str
        Follow-up session ID
    """

    print(f"\n[INFO] Preparing {tier_name}")

    out_dir = OUTPUT_ROOT / tier_name
    out_dir.mkdir(parents=True, exist_ok=True)

    session0_dir = SIMON_BIDS_ROOT / session_t0
    session1_dir = SIMON_BIDS_ROOT / session_t1

    if not session0_dir.exists():
        raise FileNotFoundError(f"Session not found: {session0_dir}")

    if not session1_dir.exists():
        raise FileNotFoundError(f"Session not found: {session1_dir}")

    t0_src = find_t1w(session0_dir)
    t1_src = find_t1w(session1_dir)

    t0_dst = out_dir / "T0.nii.gz"
    t1_dst = out_dir / "T1.nii.gz"

    shutil.copy2(t0_src, t0_dst)
    shutil.copy2(t1_src, t1_dst)

    print(f"[SUCCESS] {tier_name} created")
    print(f"  T0: {t0_src}")
    print(f"  T1: {t1_src}")


def main() -> None:
    """
    Main execution entrypoint.
    """

    print("[INFO] Preparing SIMON engineering validation datasets")

    create_test_pair(
        tier_name="simon_tier_A",
        session_t0=TIER_A_SESSION_T0,
        session_t1=TIER_A_SESSION_T1
    )

    create_test_pair(
        tier_name="simon_tier_B",
        session_t0=TIER_B_SESSION_T0,
        session_t1=TIER_B_SESSION_T1
    )

    print("\n[DONE] SIMON Tier A and Tier B datasets are ready")



if __name__ == "__main__":
    main()
