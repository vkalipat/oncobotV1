# 🩺 Medical Diagnostic Assistant - Production Grade

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (choose one)
export ANTHROPIC_API_KEY="your-key"    # RECOMMENDED: Claude Sonnet 4
# OR
export OPENAI_API_KEY="your-key"       # GPT-4o
# OR  
export GOOGLE_API_KEY="your-key"       # Gemini 2.0 Flash

# 3. The docs/ folder is already populated with 14 clinical reference PDFs

# 4. Launch
streamlit run app.py
```

## What's Included

### 📚 Medical Reference Library (docs/ folder)
14 comprehensive clinical reference PDFs covering:

| # | Document | Content |
|---|----------|---------|
| 1 | CAP Pneumonia Guidelines | CURB-65, PSI, empiric antibiotics, ICU criteria |
| 2 | UTI Guidelines | Classification, treatment by type, special populations |
| 3 | Sepsis Management | Sepsis-3 definitions, Hour-1 bundle, vasopressor protocol |
| 4 | Cardiovascular Emergencies | ACS/STEMI, HEART score, acute HF, hypertensive emergency |
| 5 | Drug Interactions & Safety | High-risk interactions, renal/hepatic dosing, contraindications |
| 6 | Emergency Differentials | Chest pain, headache, abdominal pain, dyspnea differentials |
| 7 | Diabetes Management | ADA 2024-25 guidelines, DKA vs HHS, pharmacotherapy algorithm |
| 8 | Pharmacogenomics | CYP2D6, CYP2C19, HLA alleles, warfarin PGx, DPYD |
| 9 | Neurological Emergencies | Stroke/tPA protocol, NIHSS, status epilepticus |
| 10 | Lab Interpretation | CBC, BMP, LFTs, cardiac biomarkers, inflammatory markers |
| 11 | Respiratory Guidelines | Asthma stepwise, COPD GOLD, exacerbation management |
| 12 | Endocrine Emergencies | Thyroid storm, myxedema coma, adrenal crisis |
| 13 | Infectious Disease Antibiotics | Empiric selection by site, antibiotic spectrum table |
| 14 | Psychiatric Emergencies | NMS vs serotonin syndrome, antidepressant guide, agitation |

### 🧠 Best LLM Models (2025)

| Provider | Model | Strength | USMLE Score |
|----------|-------|----------|-------------|
| **Anthropic** | Claude Sonnet 4 | Best clinical reasoning & safety | ~90% |
| **OpenAI** | GPT-4o | Highest benchmark accuracy | ~92% |
| **Google** | Gemini 2.0 Flash | Fast triage, cost-effective | ~80% |

**Recommendation**: Claude Sonnet 4 for production use (best reasoning + safety guardrails)

### 🔧 Architecture Upgrades

- **Hybrid Retrieval**: FAISS (semantic) + BM25 (lexical) for maximum recall
- **Cross-Encoder Re-ranking**: `ms-marco-MiniLM-L-6-v2` for precision after retrieval
- **Multi-Query Expansion**: Symptoms → multiple search queries (organ-system + treatment + differential)
- **5-Stage Clinical Pipeline**: Triage → Differential → Workup → Treatment → Assembly
- **Smart Cache Invalidation**: Auto-rebuilds vectorstore when docs/ contents change
- **Pharmacogenomics-Aware**: CYP2D6, CYP2C19, HLA-B*57:01 support in patient profiles

## Adding Your Own Documents

Drop additional PDFs into the `docs/` folder:
- Wellstar-specific clinical pathways
- Institutional formulary/drug references  
- UpToDate or DynaMed article exports
- NCCN oncology pathway PDFs
- Any medical reference PDFs

The vectorstore will automatically rebuild on next startup when it detects new files.

## ⚠️ Disclaimer

This is a **clinical decision support tool**, not a replacement for clinical judgment. 
All diagnostic suggestions must be reviewed by a qualified healthcare professional. 
Do not use for direct patient care without physician oversight.
