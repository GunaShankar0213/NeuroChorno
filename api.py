import uuid
import shutil
import io
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, cast

import numpy as np
import nibabel as nib
import numpy.typing as npt
import matplotlib.pyplot as plt

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Query
)

from fastapi.responses import (
    JSONResponse,
    StreamingResponse
)

from fastapi.staticfiles import StaticFiles

# Assuming this exists in your environment
from backend import run_full_pipeline
from fastapi.middleware.cors import CORSMiddleware
# ============================================================
# FastAPI App Init
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"

SESSIONS_DIR.mkdir(exist_ok=True)

# Serve sessions folder for browser access
app.mount(
    "/sessions",
    StaticFiles(directory=str(SESSIONS_DIR)),
    name="sessions"
)

# ============================================================
# SESSION MANAGEMENT
# ============================================================

def generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"SESSION_{timestamp}_{short_uuid}"

def save_upload(file: UploadFile, destination: Path):
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

# ============================================================
# RUN PIPELINE ENDPOINT
# ============================================================

@app.post("/run_pipeline")
async def run_pipeline(
    t0_file: UploadFile = File(...),
    t1_file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    interval_days: float = Form(...)
):
    try:
        session_id = generate_session_id()
        session_dir = SESSIONS_DIR / session_id
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        t0_path = input_dir / "T0.nii.gz"
        t1_path = input_dir / "T1.nii.gz"

        save_upload(t0_file, t0_path)
        save_upload(t1_file, t1_path)

        result = run_full_pipeline(
            session_dir=session_dir,
            age=age,
            sex=sex,
            interval_days=interval_days
        )

        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "session_dir": str(session_dir).replace("\\", "/"),
            "result": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# NIFTI FILE RESOLUTION
# ============================================================

def load_nifti_for_session(session_id: str, which: str) -> Optional[Path]:
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return None

    candidates_map = {
        "t0": ["input/T0.nii.gz", "module1/02_bias_corrected/T0.nii.gz", "T0.nii.gz"],
        "t1": ["input/T1.nii.gz", "module1/02_bias_corrected/T1.nii.gz", "T1.nii.gz"],
        "warped": ["module1/04_ants_syn/warped_ants.nii.gz", "module1/04_ants_syn/warp_ants.nii.gz"],
        "jacobian": ["module1/04_ants_syn/jacobian_ants.nii.gz"]
    }

    candidates = candidates_map.get(which, [])
    for rel in candidates:
        p = session_dir / rel
        if p.exists():
            return p

    for rel in candidates:
        matches = list(session_dir.rglob(Path(rel).name))
        if matches:
            return matches[0]
    return None

# ============================================================
# SLICE INFO ENDPOINT
# ============================================================

@app.get("/slice_info")
def slice_info(session: str):
    session_dir = SESSIONS_DIR / session
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    info = {}
    for key in ["t0", "t1", "warped", "jacobian"]:
        path = load_nifti_for_session(session, key)
        if path:
            # Cast to Any to prevent Pylance "load is not exported" / "get_data_shape" errors
            img: Any = nib.load(str(path))
            shape = tuple(int(x) for x in img.shape) # Use img.shape directly
            info[key] = {
                "shape": shape,
                "path": str(path).replace("\\", "/")
            }
        else:
            info[key] = None

    return JSONResponse(info)

# ============================================================
# SLICE RENDERING
# ============================================================

def extract_slice(data: npt.NDArray[Any], plane: str, index: int) -> npt.NDArray[Any]:
    if plane == "axial":
        return np.rot90(data[:, :, index])
    elif plane == "coronal":
        return np.rot90(data[:, index, :])
    elif plane == "sagittal":
        return np.rot90(data[index, :, :])
    else:
        raise ValueError("Invalid plane")

def render_png(slice_img: npt.NDArray[Any]) -> io.BytesIO:
    vmin = np.percentile(slice_img, 2)
    vmax = np.percentile(slice_img, 98)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))

    ax.imshow(slice_img, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

# ============================================================
# SLICE PNG ENDPOINT
# ============================================================

@app.get("/slice_png")
def slice_png(
    session: str,
    vol: str = Query("t0"),
    plane: str = Query("axial"),
    index: Optional[int] = None,
    overlay_jacobian: bool = False
):
    path = load_nifti_for_session(session, vol)

    # fallback logic
    if path is None:
        if vol == "warped":
            path = load_nifti_for_session(session, "t1")
            if path is None:
                raise HTTPException(status_code=404, detail="Neither warped nor t1 found")
        else:
            raise HTTPException(status_code=404, detail=f"{vol} not found")


    # Use Any to bypass nibabel's inconsistent type stubs
    img: Any = nib.load(str(path))
    data = cast(npt.NDArray[np.float64], img.get_fdata())

    # Ensure index is an integer and not None
    if plane == "axial":
        max_idx = data.shape[2] - 1
    elif plane == "coronal":
        max_idx = data.shape[1] - 1
    elif plane == "sagittal":
        max_idx = data.shape[0] - 1
    else:
        raise HTTPException(status_code=400, detail="Invalid plane")

    safe_index = index if index is not None else max_idx // 2
    safe_index = max(0, min(safe_index, max_idx))


    # Always define slice_img first
    slice_img: npt.NDArray[Any] = extract_slice(data, plane, safe_index)

    # Jacobian overlay logic (proper expansion / contraction visualization)
    if overlay_jacobian:
        jac_path = load_nifti_for_session(session, "jacobian")

        if jac_path:
            jac_img: Any = nib.load(str(jac_path))
            jac_data = cast(npt.NDArray[np.float64], jac_img.get_fdata())

            jac_slice = extract_slice(jac_data, plane, safe_index)

            # Log transform improves visualization (medical standard)
            jac_log = np.log(jac_slice + 1e-6)

            # Normalize centered at zero
            vmax = np.percentile(np.abs(jac_log), 98)
            vmin = -vmax

            jac_norm = (jac_log - vmin) / (vmax - vmin + 1e-9)
            jac_norm = np.clip(jac_norm, 0, 1)

            cmap = plt.get_cmap("bwr")
            jac_rgb = cmap(jac_norm)[:, :, :3]

            # Normalize base image
            base_vmin = np.percentile(slice_img, 2)
            base_vmax = np.percentile(slice_img, 98)

            base_norm = (slice_img - base_vmin) / (base_vmax - base_vmin + 1e-9)
            base_norm = np.clip(base_norm, 0, 1)

            base_rgb = np.stack([base_norm] * 3, axis=-1)

            alpha = 0.45

            slice_img = (1 - alpha) * base_rgb + alpha * jac_rgb


    buf = render_png(slice_img)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/compare_png")
def compare_png(
    session: str,
    plane: str = Query("axial"),
    index: Optional[int] = None,
):
    t0_path = load_nifti_for_session(session, "t0")
    t1_path = load_nifti_for_session(session, "warped") or load_nifti_for_session(session, "t1")

    if t0_path is None or t1_path is None:
        raise HTTPException(status_code=404, detail="Missing T0 or T1")

    t0_img: Any = nib.load(str(t0_path))
    t1_img: Any = nib.load(str(t1_path))

    t0 = cast(npt.NDArray[np.float64], t0_img.get_fdata())
    t1 = cast(npt.NDArray[np.float64], t1_img.get_fdata())

    if plane == "axial":
        max_idx = t0.shape[2] - 1
    elif plane == "coronal":
        max_idx = t0.shape[1] - 1
    else:
        max_idx = t0.shape[0] - 1

    safe_index = index if index is not None else max_idx // 2

    t0_slice = extract_slice(t0, plane, safe_index)
    t1_slice = extract_slice(t1, plane, safe_index)

    diff = t1_slice - t0_slice

    buf = render_png(diff)

    return StreamingResponse(buf, media_type="image/png")

@app.get("/max_slice")
def max_slice(session: str, vol: str, plane: str):

    path = load_nifti_for_session(session, vol)

    img = nib.load(str(path))
    shape = img.shape

    if plane == "axial":
        max_idx = shape[2] - 1
    elif plane == "coronal":
        max_idx = shape[1] - 1
    else:
        max_idx = shape[0] - 1

    return {"max_slice": max_idx}
