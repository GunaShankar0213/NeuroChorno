"""
Simon_Data_Extractor.py

Utility script to unwrap the SIMON dataset archive into the Data/ directory.

- Windows-safe
- Python 3.12
- Pylance / MyPy clean
- Deterministic behavior
"""

from pathlib import Path
import tarfile


def unwrap_simon_archive(
    archive_path: Path,
    output_root: Path
) -> None:
    """
    Extract the SIMON dataset tar.gz archive into Data/SIMON.

    Parameters
    ----------
    archive_path : Path
        Absolute path to SIMON_data.tar.gz
    output_root : Path
        Root directory where SIMON/ will be created

    Raises
    ------
    FileNotFoundError
        If the archive does not exist
    RuntimeError
        If extraction fails
    """

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if archive_path.suffixes[-2:] != [".tar", ".gz"]:
        raise RuntimeError("Input file is not a .tar.gz archive")

    simon_dir: Path = output_root / "SIMON"
    simon_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Extracting SIMON dataset")
    print(f"[INFO] Source : {archive_path}")
    print(f"[INFO] Target : {simon_dir}")

    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            members = tar.getmembers()
            if not members:
                raise RuntimeError("Archive is empty")

            tar.extractall(path=simon_dir)

    except Exception as exc:
        raise RuntimeError(f"Failed to extract SIMON dataset: {exc}") from exc

    print("[SUCCESS] SIMON dataset extracted successfully")


def main() -> None:
    """
    Entry point.
    """

    archive_path: Path = Path(
        r"C:\Users\gunas_ybbhfcb\Downloads\Datasets\SIMON_data.tar.gz"
    )

    output_root: Path = Path("Data")

    unwrap_simon_archive(
        archive_path=archive_path,
        output_root=output_root
    )


if __name__ == "__main__":
    main()
