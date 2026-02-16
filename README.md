# Neuro-Metric --- Hybrid Morphometry Engine for Clinical Trial Enrichment

Neuro-Metric is a hybrid physics-guided and transformer-based
morphometry system that converts longitudinal MRI scans into explainable
fast-progressor classifications for Alzheimer's clinical trial
enrichment.

------------------------------------------------------------------------

# System Overview

![System Overview](docs/images/system_overview.png)

Neuro-Metric consists of three modules:

1.  Hybrid Morphometry Engine
2.  Quantitative + MedGemma Reasoning Engine
3.  Clinical Trial Enrichment Dashboard

------------------------------------------------------------------------

# MODULE 1 --- Hybrid Morphometry Engine

![Module 1 Pipeline](docs/images/module1_pipeline.png)

## Purpose

Module 1 converts two MRI scans into a voxel-level 3D Difference Map
(Jacobian Map) that quantifies brain tissue expansion and contraction.

------------------------------------------------------------------------

## Pipeline Steps

### Step 1 --- Skull Stripping

![Skull Stripping](docs/images/skull_strip.png)

Tool: HD-BET\
Output: Brain-only MRI

------------------------------------------------------------------------

### Step 2 --- Bias Field Correction

![Bias Correction](docs/images/bias_correction.png)

Tool: N4ITK\
Output: Intensity-normalized MRI

------------------------------------------------------------------------

### Step 3 --- Affine Registration

![Affine Alignment](docs/images/affine_alignment.png)

Tool: ANTs\
Output: Affine-aligned MRI

------------------------------------------------------------------------

### Step 4 --- Transformer Registration

![TransMorph Registration](docs/images/transmorph_registration.png)

Tool: TransMorph-Large\
Output: Deformation field

------------------------------------------------------------------------

### Step 5 --- Quality Gate

![Quality Gate](docs/images/quality_gate.png)

Metrics:

-   Negative Jacobians
-   NCC similarity
-   Deformation magnitude

Fallback to ANTs SyN if DL fails.

------------------------------------------------------------------------

### Step 6 --- Jacobian Map

![Jacobian Map](docs/images/jacobian_map.png)

Output:

-   Difference Map (.nii.gz)
-   Heatmap (.png)

------------------------------------------------------------------------

# MODULE 2 --- Quantitative + MedGemma Reasoning Engine

![Module 2 Pipeline](docs/images/module2_pipeline.png)

## Steps

### ROI Quantification

![ROI Quantification](docs/images/roi_quantification.png)

Computes regional volume changes.

------------------------------------------------------------------------

### Z-Score Normalization

![Z Score](docs/images/zscore_chart.png)

Normalizes against healthy aging.

------------------------------------------------------------------------

### Fast Progressor Classification

![Progression Score](docs/images/progression_score.png)

Rule-based scoring system.

------------------------------------------------------------------------

### MedGemma Interpretation

![MedGemma Interpretation](docs/images/medgemma_interpretation.png)

Constrained explainable AI output.

------------------------------------------------------------------------

# MODULE 3 --- Clinical Trial Enrichment Dashboard

![Dashboard](docs/images/dashboard.png)

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

NeuroMetric/ ├── Modules/ ├── backend/ ├── frontend/ ├── docs/images/
└── README.md

------------------------------------------------------------------------

# Summary

Neuro-Metric combines transformer-based deformable registration,
deterministic morphometry, and constrained medical AI to produce
explainable clinical trial enrichment insights.
