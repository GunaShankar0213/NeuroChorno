"""
STEP 5 — High-Clarity T1 Axial Slice Exporter

Improved visual fidelity:
- No interpolation smoothing
- Brain-masked normalization
- Higher DPI
- Larger canvas
- No artificial filtering
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def export_t1_axial_slice(
    t1_nifti_path: Path,
    output_png_path: Path
) -> Path:

    if not t1_nifti_path.exists():
        raise FileNotFoundError(f"T1 NIfTI not found: {t1_nifti_path}")

    img = nib.load(str(t1_nifti_path))
    data = img.get_fdata().astype(np.float32)

    # Use axial mid-slice
    axial_index = data.shape[2] // 2
    slice_2d = data[:, :, axial_index]

    # --------------------------------------------------
    # Brain-only normalization (avoid background skew)
    # --------------------------------------------------

    brain_mask = slice_2d > np.percentile(slice_2d, 20)
    brain_values = slice_2d[brain_mask]

    if brain_values.size == 0:
        raise ValueError("Brain mask failed — slice empty.")

    vmin = np.percentile(brain_values, 1)
    vmax = np.percentile(brain_values, 99)

    slice_2d = np.clip(slice_2d, vmin, vmax)
    slice_2d = (slice_2d - vmin) / (vmax - vmin)

    # Rotate for anatomical correctness
    slice_2d = np.rot90(slice_2d)

    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # High resolution export
    # --------------------------------------------------

    plt.figure(figsize=(8, 8))  # Larger canvas
    plt.imshow(
        slice_2d,
        cmap="gray",
        interpolation="nearest"  # No smoothing
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_png_path,
        dpi=300,                  # High DPI
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    return output_png_path
