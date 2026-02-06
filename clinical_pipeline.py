"""
3-stage clinical reasoning pipeline and single-shot fallback.

Optimized from the original 5-stage pipeline to cut token usage ~40%
by merging Triage+Differential and Workup+Treatment into single calls,
eliminating redundant context transmission.
"""

from .prompts import (
    TRIAGE_PROMPT,
    DIFFERENTIAL_PROMPT,
    WORKUP_PROMPT,
    TREATMENT_PROMPT,
    ASSEMBLY_PROMPT,
    TRIAGE_DIFFERENTIAL_PROMPT,
    WORKUP_TREATMENT_PROMPT,
)


class ClinicalPipeline:
    """Runs the 3-stage clinical reasoning pipeline or single-shot diagnosis."""

    def __init__(self, llm):
        self.llm = llm

    def multi_stage_diagnose(self, symptoms: str, patient_str: str, doc_context: str, web_context: str) -> str:
        """3-stage clinical pipeline: Triage+Differential, Workup+Treatment, Assembly.

        Signature is identical to the original 5-stage version for drop-in compatibility.
        """

        # Stage 1: Triage + Differential (only place doc_context is sent)
        triage_diff_prompt = TRIAGE_DIFFERENTIAL_PROMPT.format(
            patient_info=patient_str, symptoms=symptoms,
            context=doc_context, web_context=web_context or "None"
        )
        triage_differential = self.llm.invoke(triage_diff_prompt).content

        # Stage 2: Workup + Treatment (uses differential output, no doc_context)
        workup_treat_prompt = WORKUP_TREATMENT_PROMPT.format(
            patient_info=patient_str, differential=triage_differential
        )
        workup_treatment = self.llm.invoke(workup_treat_prompt).content

        # Stage 3: Assembly (unchanged — synthesizes all outputs)
        assembly_prompt = ASSEMBLY_PROMPT.format(
            patient_info=patient_str, triage=triage_differential,
            differential=triage_differential, workup=workup_treatment,
            treatment=workup_treatment
        )
        final = self.llm.invoke(assembly_prompt).content

        return final

    def single_shot_diagnose(self, symptoms: str, patient_str: str, doc_context: str, web_context: str) -> str:
        """Single-shot diagnosis for faster response (fallback mode)."""
        simple_prompt = f"""You are an expert medical diagnostic assistant. Analyze the patient's symptoms and provide a comprehensive assessment.

## PATIENT PROFILE
{patient_str}

## CURRENT SYMPTOMS
{symptoms}

## MEDICAL REFERENCE CONTEXT
{doc_context}

## ADDITIONAL RESEARCH
{web_context or "None"}

Provide: differential diagnosis (ranked with percentages), recommended tests (prioritized table),
treatment plan (personalized to patient, checking allergies and interactions), and patient-specific warnings.
"""
        return self.llm.invoke(simple_prompt).content
