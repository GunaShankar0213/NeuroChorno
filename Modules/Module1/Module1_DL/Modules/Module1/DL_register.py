import os
import sys
import time
import json
import logging
import argparse

import torch
import numpy as np
import SimpleITK as sitk

import nibabel as nib
from nibabel import Nifti1Image  # type: ignore


# ---------------- LOGGER ----------------

def setup_logger():
    logger = logging.getLogger("DL_REGISTER")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        logger.addHandler(ch)

    return logger


logger = setup_logger()


# ---------------- DEVICE ----------------

def get_device():

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()

        # sm_120+ GPUs not supported by many torch wheels yet
        if cap[0] >= 12:
            logger.warning(
                f"GPU compute capability {cap} not supported by this PyTorch build — falling back to CPU"
            )
            return torch.device("cpu")

        logger.info("Using CUDA GPU")
        return torch.device("cuda")

    logger.warning("CUDA not available — using CPU")
    return torch.device("cpu")



# ---------------- RESAMPLE ----------------

def resample_to_model_size(in_path, out_path, size):

    img = sitk.ReadImage(in_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(list(size))
    resampler.SetInterpolator(sitk.sitkLinear)

    spacing = img.GetSpacing()
    orig_size = img.GetSize()

    new_spacing = [
        spacing[i] * orig_size[i] / size[i]
        for i in range(3)
    ]

    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())

    out = resampler.Execute(img)
    sitk.WriteImage(out, out_path)

    logger.info(f"Resampled → {size}")

    return out_path


# ---------------- LOAD NIFTI ----------------

def load_norm_nifti(path):

    nii = nib.load(path)  # type: ignore
    data = nii.get_fdata().astype(np.float32) #type: ignore

    data = (data - data.mean()) / (data.std() + 1e-6)

    logger.debug(f"{path} shape={data.shape}")

    return data, nii


# ---------------- MODEL LOAD ----------------

def load_transmorph(model_path, device):

    model_dir = os.path.abspath(
        "TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/models"
    )

    sys.path.insert(0, model_dir)

    from TransMorph import TransMorph  # type: ignore
    import configs_TransMorph as cfg  # type: ignore

    config = cfg.get_3DTransMorphLarge_config()

    model = TransMorph(config).to(device)

    logger.info("Loading TransMorph-Large weights")

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    logger.info("Model ready")

    return model, config


# ---------------- MAIN REGISTRATION ----------------

def run_dl_registration(
    fixed_path,
    moving_path,
    model_path,
    out_dir
):

    os.makedirs(out_dir, exist_ok=True)

    device = get_device()

    model, config = load_transmorph(model_path, device)

    fixed_rs = os.path.join(out_dir, "fixed_rs.nii.gz")
    moving_rs = os.path.join(out_dir, "moving_rs.nii.gz")

    resample_to_model_size(fixed_path, fixed_rs, config.img_size)
    resample_to_model_size(moving_path, moving_rs, config.img_size)

    fixed, fixed_nii = load_norm_nifti(fixed_rs)
    moving, _ = load_norm_nifti(moving_rs)

    fixed_t = torch.from_numpy(fixed)[None, None].to(device)
    moving_t = torch.from_numpy(moving)[None, None].to(device)

    x = torch.cat([moving_t, fixed_t], dim=1)

    logger.info(f"Input tensor → {tuple(x.shape)}")

    torch.cuda.empty_cache()

    start = time.time()

    with torch.no_grad():
        warped, flow = model(x)

    runtime = time.time() - start

    logger.info(f"Inference time {runtime:.2f}s")

    warped_np = warped.cpu().numpy()[0, 0]
    flow_np = flow.cpu().numpy()[0]

    warped_path = os.path.join(out_dir, "warped_dl.nii.gz")
    flow_path = os.path.join(out_dir, "flow_dl.npy")

    nib.save(Nifti1Image(warped_np, fixed_nii.affine), warped_path)  # type: ignore
    np.save(flow_path, flow_np)

    disp = np.linalg.norm(flow_np, axis=0)

    metrics = {
        "runtime_sec": runtime,
        "max_disp": float(disp.max()),
        "mean_disp": float(disp.mean()),
        "device": str(device)
    }

    with open(os.path.join(out_dir, "dl_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("DL registration complete")

    return warped_path, flow_path


# ---------------- CLI ----------------

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--moving", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    run_dl_registration(
        args.fixed,
        args.moving,
        args.model,
        args.out
    )
