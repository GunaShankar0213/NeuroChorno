import numpy as np
import nibabel as nib


def flow_to_nifti(flow_path, ref_img_path, out_path):

    flow = np.load(flow_path)          # (3, D, H, W)
    ref = nib.load(ref_img_path)

    flow = np.moveaxis(flow, 0, -1)    # → (D,H,W,3)

    nii = nib.Nifti1Image(flow, ref.affine, ref.header)
    nib.save(nii, out_path)

    print("Saved warp field:", out_path)
    return out_path
