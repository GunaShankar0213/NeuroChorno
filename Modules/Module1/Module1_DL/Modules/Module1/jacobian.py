import nibabel as nib
import numpy as np
from scipy.ndimage import sobel


def compute_jacobian_determinant(warp_path, out_path, log_jacobian=True):
    """
    Compute Jacobian determinant from displacement warp field (NIfTI vector image).
    warp_path : (D,H,W,3) displacement field
    """

    nii = nib.load(warp_path)
    flow = nii.get_fdata()   # (D,H,W,3)

    ux = flow[..., 0]
    uy = flow[..., 1]
    uz = flow[..., 2]

    # --- spatial gradients ---
    dux_dx = sobel(ux, axis=0) / 8.0
    dux_dy = sobel(ux, axis=1) / 8.0
    dux_dz = sobel(ux, axis=2) / 8.0

    duy_dx = sobel(uy, axis=0) / 8.0
    duy_dy = sobel(uy, axis=1) / 8.0
    duy_dz = sobel(uy, axis=2) / 8.0

    duz_dx = sobel(uz, axis=0) / 8.0
    duz_dy = sobel(uz, axis=1) / 8.0
    duz_dz = sobel(uz, axis=2) / 8.0

    # --- Jacobian matrix components ---
    Jxx = 1 + dux_dx
    Jxy = dux_dy
    Jxz = dux_dz

    Jyx = duy_dx
    Jyy = 1 + duy_dy
    Jyz = duy_dz

    Jzx = duz_dx
    Jzy = duz_dy
    Jzz = 1 + duz_dz

    # --- determinant ---
    detJ = (
        Jxx * (Jyy * Jzz - Jyz * Jzy)
        - Jxy * (Jyx * Jzz - Jyz * Jzx)
        + Jxz * (Jyx * Jzy - Jyy * Jzx)
    )

    if log_jacobian:
        detJ = np.log(np.clip(detJ, 1e-6, None))

    out_img = nib.Nifti1Image(detJ.astype(np.float32), nii.affine, nii.header)
    nib.save(out_img, out_path)

    print("Saved Jacobian map:", out_path)

    return out_path
