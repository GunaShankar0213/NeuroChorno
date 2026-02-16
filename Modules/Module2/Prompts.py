"""
Prompt Builder — MedGemma 1.5 Optimized.
Implements the "Constrained Intelligence Layer" for Module 2.
Enforces deterministic numeric handling while allowing rich functional interpretation.
"""

import json
from typing import Dict, Any

def _constraint_block() -> str:
    return """
### SYSTEM ROLE
You are an expert Neuro-Morphometry Analyst assisting in a clinical trial enrichment protocol.
Your goal is to interpret quantitative brain volume changes (Jacobian determinants) relative to age-matched norms.

### CORE CONSTRAINTS (NON-NEGOTIABLE)
1. **ZERO HALLUCINATION:** You must use ONLY the numeric values provided in the "Quantitative Data" section.
2. **NUMERIC INTEGRITY:** Never round, alter, or invent ROI metrics. Copy Annual % Change and Z-scores exactly as they appear.
3. **NO DIAGNOSIS:** Do not diagnose "Alzheimer's Disease" or "MCI". Use descriptive terms like "neurodegeneration pattern" or "atrophic progression".
4. **DETERMINISTIC CLASSIFICATION:** You must respect the provided "Fast Progressor" score. Do not re-calculate it.
5. **FUNCTIONAL FOCUS:** Connect structural changes to their functional implications (e.g., Hippocampus -> Memory), but remain observational.
"""

# --------------------------------------------------------------------------
# STAGE 1 — Quantitative Interpretation (Text)
# Focus: Magnitude assessment and pattern recognition.
# --------------------------------------------------------------------------
def build_numeric_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    roi = payload["roi_metrics"]
    progression = payload["progression"]
    
    # Format ROI data for clarity
    roi_text = "\n".join([
        f"- {r.replace('_', ' ').title()}: {val['annual_percent_change']}%/year (Z-Score: {val['z_score']})"
        for r, val in roi.items()
    ])

    prompt = f"""
{_constraint_block()}

### STAGE 1: QUANTITATIVE DATA ANALYSIS

### PATIENT CONTEXT
- Age: {payload["metadata"]["age"]}
- Sex: {payload["metadata"]["sex"]}
- Scan Interval: {payload["metadata"]["interval_years"]} years

### QUANTITATIVE DATA (Source of Truth)
{roi_text}

### AUTOMATED CLASSIFICATION
- Class: {progression["class"]}
- Score: {progression["score"]}
- Rules Triggered: {'; '.join(progression['rationale'])}

### TASK
Write a concise analytical summary (plain text) that:
1. **Evaluates Severity:** Identify which regions show "significant deviation" (typically Z < -1.5).
2. **Pattern Recognition:** Is the atrophy global (all regions) or focal (specific regions like Entorhinal Cortex)?
3. **Progression Context:** Explain why the automated classification (Normal vs. Fast) makes sense based on the numbers provided.

### OUTPUT
(Provide a 1-paragraph analytical summary).
"""
    return {"text": prompt, "images": [], "stage": "stage1"}


# --------------------------------------------------------------------------
# STAGE 2 — Multimodal Integration (Text)
# Focus: Visual sanity check (Do the numbers match the picture?).
# --------------------------------------------------------------------------
def build_multimodal_prompt(payload: Dict[str, Any], stage1_output: str) -> Dict[str, Any]:
    prompt = f"""
{_constraint_block()}

### STAGE 2: MULTIMODAL VISUAL VALIDATION

### PREVIOUS ANALYSIS (Stage 1)
"{stage1_output}"

### VISUAL INPUTS
1. **Follow-up T1 Slice:** Anatomical reference.
2. **Jacobian Overlay:** Heatmap of tissue change (Blue/Cool = Atrophy, Red/Warm = Expansion).

### TASK
Compare the Quantitative Data from Stage 1 against the Visual Inputs provided here.
1. **Spatial Concordance:** Does the Jacobian overlay show "cool/blue" colors in the regions where we measured negative Z-scores (e.g., Hippocampus)?
2. **Artifact Check:** Are there any obvious visual anomalies that contradict the numbers?
3. **Conclusion:** State whether the visual evidence *supports* or *contradicts* the quantitative metrics.

### OUTPUT
(Provide a short paragraph focusing on visual-numeric alignment).
"""
    # Context images are critical here
    return {
        "text": prompt, 
        "images": [
            payload["context_images"]["t1_followup_slice"], 
            payload["context_images"]["jacobian_overlay"]
        ], 
        "stage": "stage2"
    }


# --------------------------------------------------------------------------
# STAGE 3 — Self Verification (Text)
# Focus: Error correction before final formatting.
# --------------------------------------------------------------------------
def build_verification_prompt(payload: Dict[str, Any], stage1_output: str, stage2_output: str) -> Dict[str, Any]:
    prompt = f"""
{_constraint_block()}

### STAGE 3: LOGIC & CONSISTENCY CHECK

### INPUTS
- Stage 1 (Numbers): {stage1_output}
- Stage 2 (Visuals): {stage2_output}

### TASK
Act as a Quality Assurance Auditor. Review the findings above.
1. Did the analysis stick to the provided ROI numbers exactly?
2. Did the analysis avoid diagnosing specific diseases?
3. Is the functional interpretation logical (e.g., temporal lobe atrophy linked to memory/language)?

If you find any inconsistencies, correct them now. If everything is accurate, summarize the validated clinical narrative.

### OUTPUT
(Provide a corrected, finalized narrative summary).
"""
    return {"text": prompt, "images": [], "stage": "stage3"}


# --------------------------------------------------------------------------
# STAGE 4 — Final Clinical Narrative (JSON)
# Focus: Structured data generation for the pipeline.
# --------------------------------------------------------------------------
def build_simplification_prompt(payload: Dict[str, Any], stage3_output: str) -> Dict[str, Any]:
    roi = payload["roi_metrics"]
    progression = payload["progression"]

    # Explicit schema definition for the model
    # Note: We pre-fill the numeric values in the example to show the model 
    # that "exact copying" is the expected behavior.
    schema_example = {
        "roi_interpretations": {
            "hippocampus": {
                "annual_percent_change": -1.5,
                "z_score": -0.5,
                "interpretation": "Brief functional comment on this specific region."
            },
            # ... (repeat for all ROIs)
        },
        "final_narrative": "A cohesive clinical paragraph summarizing the findings...",
        "classification": {
            "class": progression["class"],
            "score": progression["score"],
            "rationale": progression["rationale"]
        },
        "confidence_level": "High",
        "warning_flag": None
    }

    prompt = f"""
{_constraint_block()}

### STAGE 4: FINAL STRUCTURED REPORT GENERATION

### CONTEXT
You have validated the data in previous stages. Now, you must format it for the Clinical Trial Database.

### INPUT DATA (Source of Truth)
{json.dumps(roi, indent=2)}

### VERIFIED NARRATIVE (From Stage 3)
"{stage3_output}"

### INSTRUCTIONS
Generate a JSON object that matches the schema below perfectly.
1. **ROI Interpretations:** For each ROI, copy the `annual_percent_change` and `z_score` EXACTLY from the Input Data. Write a specific 1-sentence `interpretation` for that region based on the Z-score magnitude.
2. **Final Narrative:** Synthesize the "Verified Narrative" into a professional clinical summary.
3. **Classification:** Copy the `classification` object exactly as provided below:
   {json.dumps(progression, indent=2)}
4. **Confidence:** Set to "High" unless visual-numeric mismatch was noted in Stage 2.

### JSON SCHEMA EXAMPLE
{json.dumps(schema_example, indent=2)}

### OUTPUT
Return ONLY the valid JSON object. No markdown formatting, no preambles.
"""
    return {"text": prompt, "images": [], "stage": "stage4"}