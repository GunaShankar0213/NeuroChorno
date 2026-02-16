import nibabel as nib
import numpy as np
from pathlib import Path

def inspect(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    unique = np.unique(data)

    print(f"\nAtlas: {path.name}")
    print(f"Shape: {data.shape}")
    print(f"Number of unique labels: {len(unique)}")
    print(f"Labels:")
    print(unique)

# ---- CHANGE THESE PATHS ----
cort_path = Path("module2/atlas/harvard_oxford_cortical.nii.gz")
sub_path = Path("module2/atlas/harvard_oxford_subcortical.nii.gz")


inspect(cort_path)
inspect(sub_path)
