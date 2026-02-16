from nilearn import datasets
import nibabel as nib
import json
from pathlib import Path

# Fetch atlas
cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')

target_dir = Path("module2/atlas")
target_dir.mkdir(parents=True, exist_ok=True)

# Save atlas maps
nib.save(cort.maps, target_dir / "harvard_oxford_cortical.nii.gz")
nib.save(sub.maps, target_dir / "harvard_oxford_subcortical.nii.gz")

# Save labels
with open(target_dir / "harvard_oxford_cortical_labels.json", "w") as f:
    json.dump(cort.labels, f, indent=2)

with open(target_dir / "harvard_oxford_subcortical_labels.json", "w") as f:
    json.dump(sub.labels, f, indent=2)

print("Atlas + labels saved to:", target_dir.resolve())
