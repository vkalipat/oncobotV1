"""
Multi-query expansion from symptoms for better document recall.
"""

from typing import List


class QueryExpander:
    """Generates multiple search queries from symptoms for better recall."""

    SYSTEM_KEYWORDS = {
        "respiratory": ["cough", "sputum", "dyspnea", "breath", "wheez", "chest", "pneumon", "bronch"],
        "cardiovascular": ["chest pain", "palpitat", "hypertens", "blood pressure", "tachycard", "bradycard", "syncope", "edema"],
        "gastrointestinal": ["nausea", "vomit", "diarrhea", "abdomin", "bowel", "constipat", "dysphagia", "jaundice"],
        "neurological": ["headache", "dizz", "seizure", "numbness", "weakness", "confus", "altered mental", "stroke"],
        "urinary": ["dysuria", "hematuria", "frequen", "urinat", "flank pain", "suprapubic"],
        "musculoskeletal": ["joint", "pain", "stiffness", "swelling", "arthri", "back pain"],
        "dermatological": ["rash", "lesion", "itch", "skin", "erythem", "pustul"],
        "infectious": ["fever", "chills", "rigors", "sepsis", "infection"],
        "endocrine": ["thirst", "polyuria", "weight loss", "weight gain", "fatigue", "thyroid", "diabet"],
        "psychiatric": ["anxiety", "depression", "insomnia", "agitat", "hallucinat", "suicid"],
    }

    def expand_medical_queries(self, symptoms: str) -> List[str]:
        """Generate multiple search queries from symptoms for better recall."""
        queries = [symptoms]

        symptom_lower = symptoms.lower()

        # Add organ-system focused queries
        for system, keywords in self.SYSTEM_KEYWORDS.items():
            if any(kw in symptom_lower for kw in keywords):
                queries.append(f"{system} {symptoms[:100]}")

        # Add treatment-focused query
        queries.append(f"treatment {symptoms[:100]}")

        # Add differential diagnosis query
        queries.append(f"differential diagnosis {symptoms[:100]}")

        return queries[:5]  # Cap at 5 queries
