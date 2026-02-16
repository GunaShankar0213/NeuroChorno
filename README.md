# Neuro-Chrono - Morphometry Engine for Clinical Trial Enrichment

Neuro-Chrono is a hybrid physics-guided and transformer-inspired morphometry system that converts longitudinal brain MRI scans into explainable, quantitative progression metrics and fast-progressor classifications for Alzheimer's clinical trial enrichment.

It combines diffeomorphic registration, deterministic morphometry, and constrained medical AI reasoning to produce clinically interpretable and scientifically defensible outputs.

"The core technology—"3D Longitudinal Registration + MedGemma Difference Analysis"—works for any solid organ that changes shape over time."

View this : [Click this](https://drive.google.com/file/d/1n6hJoikn59r0rQjagH6NwOm5yufmgtcj/view?usp=sharing)

------------------------------------------------------------------------

# Key Features

• Clinical-grade deformable registration using ANTs SyN
• Jacobian determinant-based morphometry (scientific gold standard)
• Deterministic ROI quantification and Z-score normalization
• Constrained MedGemma medical reasoning (non-hallucinatory)
• Fast-progressor classification for clinical trial enrichment
• Explainable outputs with numeric and visual evidence
• Fully automated end-to-end pipeline

------------------------------------------------------------------------
# System Overview

<img width="2816" height="1536" alt="architecture" src="https://github.com/user-attachments/assets/935dcde5-e4be-459c-9939-41d391302aec" />


Neuro-Chrono consists of three integrated modules:

| Module   | Name                                     | Role                                                            |
| -------- | ---------------------------------------- | --------------------------------------------------------------- |
| Module 1 | Hybrid Morphometry Engine                | Computes voxel-level Difference Map (Jacobian Map) [DL/ANTs Syn]|
| Module 2 | Quantitative + MedGemma Reasoning Engine | Converts morphometry into explainable progression metrics       |
| Module 3 | Clinical Trial Enrichment Dashboard      | Provides visualization, reports, and trial eligibility insights |

How to run : [watch this](https://drive.google.com/file/d/1TpV_rDFbbnlVLTzDZ6GkFZLwIjX6KxKj/view?usp=sharing)
------------------------------------------------------------------------

# Installation

## 1. Clone repository

```bash
git clone https://github.com/GunaShankar0213/NeuroChorno.git
cd NeuroChorno
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

# MODULE 1 --- Hybrid Morphometry Engine

## Purpose

Module 1 converts two MRI scans into a voxel-level 3D Difference Map
(Jacobian Map) that quantifies brain tissue expansion and contraction.

------------------------------------------------------------------------

## Pipeline Steps

### Step 1 --- Skull Stripping

<img width="1800" height="900" alt="T1_qc_report" src="https://github.com/user-attachments/assets/392bacd8-819f-4629-bcff-b50b3e76e6d9" />

Tool: HD-BET\
Output: Brain-only MRI

------------------------------------------------------------------------

### Step 2 --- Bias Field Correction

<img width="1800" height="900" alt="T1_bet_cropped_n4" src="https://github.com/user-attachments/assets/f49715c5-7b3a-46f2-94d3-e6de6af98e2d" />


Tool: N4ITK\
Output: Intensity-normalized MRI

------------------------------------------------------------------------

### Step 3 --- Affine Registration


Tool: ANTs\
Output: Affine-aligned MRI

------------------------------------------------------------------------

### Step 4 --- Transformer Registration

<img width="2700" height="900" alt="jacobian_overlay" src="https://github.com/user-attachments/assets/2bbae21c-14a3-4c24-9034-c351e459c269" />

Tool: TransMorph-Large or ANTs Syn
Output: Deformation field

------------------------------------------------------------------------

### Step 5 --- Quality Gate

<img width="1898" height="1022" alt="warning" src="https://github.com/user-attachments/assets/1efbad1f-17fe-4079-84ef-2ea7c6e1f79a" />


Metrics:

-   Negative Jacobians
-   NCC similarity
-   Deformation magnitude

Fallback to ANTs SyN if DL fails.

------------------------------------------------------------------------

### Step 6 --- Jacobian Map

Output:

-   Difference Map (.nii.gz)
-   Heatmap (.png)

------------------------------------------------------------------------

# MODULE 2 --- Quantitative + MedGemma Reasoning Engine

<img width="1663" height="552" alt="thinking trance" src="https://github.com/user-attachments/assets/72cdb437-4f1b-4d64-aeed-000833ff1d6b" />

## Steps

### ROI Quantification

<img width="910" height="405" alt="image" src="https://github.com/user-attachments/assets/a22a84f9-9fe1-438b-b725-965eda782663" />


Computes regional volume changes.

------------------------------------------------------------------------

### Z-Score Normalization


Normalizes against healthy aging.

------------------------------------------------------------------------

### Fast Progressor Classification

Rule-based scoring system.

------------------------------------------------------------------------

### MedGemma Interpretation

Constrained explainable AI output.

------------------------------------------------------------------------

# MODULE 3 --- Clinical Trial Enrichment Dashboard

<img width="1890" height="1025" alt="dashboard" src="https://github.com/user-attachments/assets/b146ae3e-d707-4d11-a57a-75cfb3cde1d0" />

# Scientific Basis

Neuro-Chrono uses Jacobian determinant analysis, a well-established method in computational neuroanatomy for measuring local volume change.

Applications include:

• Alzheimer's disease progression analysis
• Neurodegeneration research
• Clinical morphometry
• Clinical trial recruitment

## Features

-   MRI viewer
-   Heatmap visualization
-   Fast progressor badge
-   AI explanation panel
-   Exportable reports

------------------------------------------------------------------------

# Output Files

-   jacobian_map.nii.gz
-   heatmap.png
-   roi_deltas.json
-   z_scores.json
-   progression_score.json
-   report.pdf

------------------------------------------------------------------------

# Hardware Requirements

Validated on:

-   Intel i7-14500HX
-   RTX 5060 GPU
-   32GB RAM

------------------------------------------------------------------------

# Runtime

Total pipeline runtime: 12--18 minutes

------------------------------------------------------------------------

# Repository Structure

```
NeuroChrono/
│
├── Modules/          # Core processing modules (Module-1, Module-2)
├── backend/          # Pipeline orchestration, session management, APIs
├── frontend/         # User interface (Streamlit / Web UI)
├── docs/
│   └── images/      # README images, diagrams, and visual assets
│
└── README.md        # Project documentation
```

**Note:  In the above code, only ANTS Syn is integrated, due to the DL registartion can be used transmorph as a Add-on where no open training weights is publically avaliable optimsed.**

------------------------------------------------------------------------

# Summary

Neuro-Chrono combines transformer-based deformable registration,
deterministic morphometry, and constrained medical AI to produce
explainable clinical trial enrichment insights.

------------------------------------------------------------------------

# Known Limitations

1. Skull-stripping affine metric warnings may occur due to irregular cropping.
2. MedGemma inference latency depends on hardware availability.
3. Visualization precision depends on input image resolution.

------------------------------------------------------------------------
# Future Enhancements:
- The "Liver-Chrono" (Oncology)
- The "Lung-Chrono" (Fibrosis)
- The "Kidney-Chrono" (PKD)

# License

MIT License

---

# Author

Guna Shankar S
Gen Ai Engineer

------------------------------------------------------------------------
# Citations

## OASIS-3 Dataset

This project uses data from the Open Access Series of Imaging Studies (OASIS-3).

If you use this dataset, please cite:

LaMontagne PJ, Benzinger TLS, Morris JC, Keefe S, Hornbeck R, Xiong C, Grant E, Hassenstab J, Moulder K, Vlassenko AG, Raichle ME, Cruchaga C, Marcus D.  
OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease.  
medRxiv. 2019.  

Dataset link:  
https://www.oasis-brains.org/

---

## MedGemma and HAI-DEF Models

This project uses MedGemma from Google Health AI Developer Foundations.

Please cite:

Mahvar F, Liu Y, Golden D, et al.  
The MedGemma Impact Challenge. Kaggle. 2026.  
https://kaggle.com/competitions/med-gemma-impact-challenge

---

## ANTs Registration Framework

This project uses ANTs for diffeomorphic registration.

Please cite:

Avants BB, Tustison NJ, Song G.  
Advanced normalization tools (ANTs).  
NeuroImage, 2011.  




