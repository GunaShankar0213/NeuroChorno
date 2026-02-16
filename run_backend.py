from pathlib import Path
from backend import run_full_pipeline


if __name__ == "__main__":

    session_dir = Path("Data/sessions/session_001")

    result = run_full_pipeline(
        session_dir=session_dir,
        age=62,
        sex="M",
        interval_days=417
    )

    print("Pipeline completed")
    print(result)
