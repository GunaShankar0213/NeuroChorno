import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def visualize_jacobian_overlay(
    jac_path,
    ref_path,
    out_png,
    slice_axis=2,
    slice_index=None,
    clip_value=0.5,
    alpha=0.55
):
    """
    Visualize Jacobian difference map overlay on MRI.

    Fixes included:
    - Brain mask cleanup
    - Clinically meaningful clipping
    - Stable normalization
    - Edge artifact suppression
    """

    # -------------------------
    # Load volumes
    # -------------------------
    jac = nib.load(jac_path).get_fdata()
    ref = nib.load(ref_path).get_fdata()

    # -------------------------
    # Brain mask cleanup
    # -------------------------
    brain_mask = (ref > 0).astype(np.float32)
    jac = jac * brain_mask

    # -------------------------
    # Choose slice
    # -------------------------
    if slice_index is None:
        slice_index = jac.shape[slice_axis] // 2

    if slice_axis == 0:
        jac_slice = jac[slice_index, :, :]
        ref_slice = ref[slice_index, :, :]
    elif slice_axis == 1:
        jac_slice = jac[:, slice_index, :]
        ref_slice = ref[:, slice_index, :]
    else:
        jac_slice = jac[:, :, slice_index]
        ref_slice = ref[:, :, slice_index]

    # -------------------------
    # MRI normalization (robust)
    # -------------------------
    p1, p99 = np.percentile(ref_slice, (1, 99))
    ref_slice = np.clip(ref_slice, p1, p99)
    ref_slice = (ref_slice - p1) / (p99 - p1 + 1e-6)

    # -------------------------
    # Jacobian display clipping
    # -------------------------
    jac_slice = np.clip(jac_slice, -clip_value, clip_value)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(7, 7))

    plt.imshow(ref_slice.T, cmap="gray", origin="lower")

    plt.imshow(
        jac_slice.T,
        cmap="bwr",
        alpha=alpha,
        vmin=-clip_value,
        vmax=clip_value,
        origin="lower"
    )

    plt.colorbar(label="Log Jacobian (volume change)")
    plt.title("Module-1 Morphometry Difference Map")

    plt.axis("off")

    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

    print("Saved visualization:", out_png)
