import numpy as np
import nibabel as nib
import json
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import structural_similarity as ssim


def compute_displacement_stats(flow):
    mag = np.linalg.norm(flow, axis=0)
    return float(mag.max()), float(mag.mean())


def compute_ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.mean(a * b))


def qa_check(flow_path, fixed_path, warped_path, out_json):

    flow = np.load(flow_path)  # (3, D, H, W)
    fixed = nib.load(fixed_path).get_fdata()
    warped = nib.load(warped_path).get_fdata()

    # --- displacement stats ---
    max_disp, mean_disp = compute_displacement_stats(flow)

    # --- similarity ---
    ncc_val = compute_ncc(fixed, warped)

    # --- smoothness proxy ---
    smoothness = float(gaussian_gradient_magnitude(flow[0], 1).mean())

    metrics = {
        "max_displacement": max_disp,
        "mean_displacement": mean_disp,
        "ncc_similarity": ncc_val,
        "smoothness": smoothness
    }

    # ---- decision logic ----
    accept = (
        max_disp < 40 and
        ncc_val > 0.90 and
        smoothness < 5
    )

    metrics["accepted"] = bool(accept)

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    return accept, metrics
