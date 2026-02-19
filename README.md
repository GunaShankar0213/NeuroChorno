# NeuroChrono: Agentic Morphometry Engine V2 with DL

NeuroChrono is a hybrid physics-guided and transformer-inspired morphometry system that converts longitudinal brain MRI scans into explainable, quantitative progression metrics and fast-progressor classifications for Alzheimer's clinical trial enrichment.

It combines diffeomorphic registration, deterministic morphometry, and constrained medical AI reasoning to produce clinically interpretable and scientifically defensible outputs.

The core technology—**3D Longitudinal Registration + MedGemma Difference Analysis**—enables precise measurement of structural change over time and can generalize to any solid organ undergoing morphological transformation.

📌 Demo Overview:  
[Click here to view system demo](https://drive.google.com/file/d/1n6hJoikn59r0rQjagH6NwOm5yufmgtcj/view?usp=sharing)

---

# V2 Integration Status (Development Branch)

NeuroChrono V2 introduces a modular deep learning registration framework alongside the validated ANTs SyN pipeline.

## Current Status

| Component | Status | Production Role |
|--------|--------|----------------|
| ANTs SyN Registration | Fully integrated | Primary production backend |
| TransMorph Integration | ⚠ Infrastructure ready | Training required |
| Hybrid Registration Framework | Complete | Supports pluggable backends |
| Safety Fallback System | Complete | Automatic fallback to ANTs |

---

## Critical Design Decision: ANTs as Production Backbone

Due to the absence of publicly available, clinically optimized pretrained weights, deep learning registration models currently exhibit:

- Large deformation instability
- Anatomical misalignment artifacts
- Non-physiological displacement fields
- Jacobian determinant inconsistencies

These effects create unacceptable risk in clinical and production workflows.

Training TransMorph or equivalent models to match the reliability of ANTs SyN requires:

- Massive longitudinal MRI datasets  
  (ADNI3, OASIS-3 scale)

- High-performance training infrastructure  
  (NVIDIA H100 clusters or equivalent)

- Extensive hyperparameter optimization

- Multi-week training cycles

ANTs SyN, by contrast, provides:

- Deterministic outputs
- Proven clinical validation
- Topology-preserving deformation fields
- Zero-training deployment reliability

For this reason:

**V1 uses ANTs SyN as the primary registration engine to ensure maximum morphometry accuracy.**

**V2 introduces DL registration only as an experimental augmentation, with ANTs serving as the automated safety fallback.**

This ensures the Agentic reasoning system operates only on scientifically reliable deformation data.

---

# Key Features

• Clinical-grade deformable registration using ANTs SyN  
• Hybrid registration framework supporting DL and classical methods  
• Jacobian determinant-based morphometry (scientific gold standard)  
• Deterministic ROI quantification and Z-score normalization  
• Constrained MedGemma medical reasoning (non-hallucinatory)  
• Fast-progressor classification for clinical trial enrichment  
• Explainable outputs with numeric and visual evidence  
• Fully automated end-to-end pipeline  
• Safety-validated fallback registration architecture  

---

# System Overview

![architecture](https://github.com/user-attachments/assets/e19427dd-0208-42e4-988b-ea10e816920e)

### NeuroChrono consists of three integrated modules:

| Module | Name | Role |
|------|------|------|
| Module 1 | Hybrid Morphometry Engine | Computes voxel-level deformation fields using ANTs or DL |
| Module 2 | Quantitative + MedGemma Reasoning Engine | Converts morphometry into explainable progression metrics |
| Module 3 | Clinical Trial Enrichment Dashboard | Provides visualization, reports, and eligibility insights |

📌 Installation and execution walkthrough:  
[Watch Technical Walk-through guide](https://drive.google.com/file/d/1TpV_rDFbbnlVLTzDZ6GkFZLwIjX6KxKj/view?usp=sharing)

---

# MODULE 1 — Hybrid Morphometry Engine

## Purpose

Module 1 converts two longitudinal MRI scans into a voxel-level Jacobian Difference Map representing anatomical expansion and contraction.

This module supports multiple registration backends:

| Backend | Status | Usage |
|-------|--------|------|
| ANTs SyN | Production Ready | Default |
| TransMorph | Infrastructure Ready | Requires training |

ANTs serves as automatic fallback if DL registration fails validation.

---

## Pipeline Steps (Using DL--- ForANTs Use Main)

---

### Step 1 — Skull Stripping

<img width="1800" height="900" alt="T0_qc_report" src="https://github.com/user-attachments/assets/9157f99f-12df-4be6-bd9e-099aae99a2df" />


Tool: HD-BET  
Output: Brain-only MRI volume

---

### Step 2 — Bias Field Correction

<img width="1800" height="900" alt="result_image_n4" src="https://github.com/user-attachments/assets/51a63e31-8c27-4290-ba90-11c50f1fdcc7" />


Tool: N4ITK  
Output: Intensity-normalized MRI

---

### Step 3 — Affine Registration


<img width="1500" height="1500" alt="QC_Checkerboard" src="https://github.com/user-attachments/assets/c4d223ac-4d24-494e-ae88-dd73a06f010b" />

Tool: ANTs  
Output: Spatially aligned MRI

---

### Step 4 — Deformable Registration

<img width="911" height="913" alt="jacobian_overlay" src="https://github.com/user-attachments/assets/8c0ec9f9-ae8b-4b0e-af94-6913cf19bbbd" />

Backends:

• ANTs SyN (Production)  
• TransMorph (Training required)

Output:

• Deformation field  
• Warped image  

---

### Step 5 — Registration Quality Gate

```
{
  "max_displacement": 7.054327964782715,
  "mean_displacement": 1.9687613248825073,
  "ncc_similarity": 0.9876299645398302,
  "smoothness": 0.1640746295452118,
  "accepted": true
}
```

Validation metrics:

• Negative Jacobian detection  
• Cross-correlation similarity  
• Deformation magnitude threshold  
• Anatomical plausibility verification  

Automatic fallback to ANTs SyN if DL validation fails, but here the displacement is more.

---

### Step 6 — Jacobian Map Generation

Outputs:

• jacobian_map.nii.gz  
• heatmap.png  

---

# Module 2 and Module 3 ---> Remains same as the Main Branch

# Scientific Basis

NeuroChrono uses Jacobian determinant analysis, a validated computational neuroanatomy method for measuring local volume change.

Applications include:

• Alzheimer's disease progression  
• Neurodegenerative disease monitoring  
• Clinical morphometry  
• Trial participant stratification  

---

# Output Files

Generated outputs:

```

jacobian_map.nii.gz
heatmap.png
roi_deltas.json
z_scores.json
progression_score.json
report.pdf

```

---

# Hardware Requirements

Validated configuration:

• Intel i7-14500HX  
• RTX 5060 GPU  
• 32GB RAM  

Training DL registration requires:

• NVIDIA H100 or equivalent  
• Large MRI datasets  
• Multi-day training time  
• ADNI-3,OASIS-3 dataset to Train the weights.
---

# Runtime

Typical runtime:

| Backend | Runtime |
|--------|--------|
| ANTs SyN | 10–15 minutes |
| DL Registration | <1 minute (experimental) |

---

# Repository for DL Results check

```

NeuroChrono/

├── Modules/
  ├── Module1/
    ├── Module1_DL/ "Data\outputs"
├── backend/
├── frontend/

```

---

# Known Limitations

1. DL registration requires trained weights for clinical reliability  
2. Public pretrained models may produce unstable deformation fields  
3. Visualization accuracy depends on input MRI resolution
4. Deep learning registration models currently sacrifice anatomical accuracy for speed due to the absence of clinically optimized pretrained weights. This results in lower deformation reliability, unstable Jacobian maps, and reduced suitability for scientific morphometry evaluation.
5. Achieving both high-speed registration and clinical-grade accuracy requires large-scale supervised training on longitudinal neuroimaging datasets such as ADNI3 and OASIS-3, using high-performance GPU infrastructure (e.g., NVIDIA H100 clusters).
6. Suboptimal registration quality directly impacts downstream morphometry features, which can reduce the reliability and interpretability of MedGemma reasoning outputs. Therefore, ANTs SyN remains the primary production backend, with DL registration currently limited to experimental evaluation in V2.

---
