# visualize_results.py
# View T0 (fixed), Warped T1, and Jacobian overlay (heatmap)

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data


def normalize(data):
    p1, p99 = np.percentile(data[data > 0], (1, 99))
    return np.clip((data - p1) / (p99 - p1 + 1e-8), 0, 1)


def main(t0_path, warped_t1_path, jacobian_path, out_png):
    t0 = load_nii(t0_path)
    wt1 = load_nii(warped_t1_path)
    jac = load_nii(jacobian_path)

    # pick central slice (axial)
    z = t0.shape[2] // 2

    t0_n = normalize(t0[:, :, z])
    wt1_n = normalize(wt1[:, :, z])

    # Jacobian visualization: clamp for stability
    jac_slice = jac[:, :, z]
    jac_clip = np.clip(jac_slice, 0.7, 1.3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.rot90(t0_n), cmap="gray")
    axes[0].set_title("T0 (Fixed)")
    axes[0].axis("off")

    axes[1].imshow(np.rot90(wt1_n), cmap="gray")
    axes[1].set_title("Warped T1")
    axes[1].axis("off")

    axes[2].imshow(np.rot90(t0_n), cmap="gray")
    hm = axes[2].imshow(
        np.rot90(jac_clip),
        cmap="coolwarm",
        alpha=0.6,
        vmin=0.7,
        vmax=1.3,
    )
    axes[2].set_title("Jacobian Overlay (Blue=Shrink, Red=Expand)")
    axes[2].axis("off")

    cbar = fig.colorbar(hm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Jacobian Determinant")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"Saved visualization to: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", required=True, help="Fixed image (T0)")
    parser.add_argument("--warped", required=True, help="Warped T1 image")
    parser.add_argument("--jacobian", required=True, help="Jacobian determinant image")
    parser.add_argument("--out", default="jacobian_overlay.png", help="Output PNG")
    args = parser.parse_args()

    main(args.t0, args.warped, args.jacobian, args.out)
