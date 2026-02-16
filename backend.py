# backend.py

from pathlib import Path
from Modules.Module1.Module1_orchestrator import run_module1
from Modules.Module2.Module2_orchestrator import run_module2


def run_full_pipeline(session_dir: Path, age: int, sex: str, interval_days: float):

    input_dir = session_dir / "input"
    module1_dir = session_dir / "module1"
    module2_dir = session_dir / "module2"

    module1_dir.mkdir(parents=True, exist_ok=True)
    module2_dir.mkdir(parents=True, exist_ok=True)

    t0 = input_dir / "T0.nii.gz"
    t1 = input_dir / "T1.nii.gz"

    if not t0.exists() or not t1.exists():
        raise FileNotFoundError("T0 or T1 image missing in session input folder")

    # -------------------
    # Run Module-1
    # -------------------
    run_module1(
        t0_path=t0,
        t1_path=t1,
        workdir=module1_dir
    )

    # Module-1 outputs
    jacobian = module1_dir / "04_ants_syn/jacobian_ants.nii.gz"
    t0_n4 = module1_dir / "02_bias_corrected/T0/T0_bet_cropped_n4.nii.gz"
    t1_n4 = module1_dir / "02_bias_corrected/T1/T1_bet_cropped_n4.nii.gz"
    overlay = module1_dir / "05_visualization/jacobian_overlay.png"

    # -------------------
    # Run Module-2
    # -------------------
    result = run_module2(
        jacobian_path=jacobian,
        t0_path=t0_n4,
        t1_followup_path=t1_n4,
        jacobian_overlay_path=overlay,
        age=age,
        sex=sex,
        interval_days=interval_days,
        output_dir=module2_dir
    )

    return result
