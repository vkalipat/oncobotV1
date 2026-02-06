"""
Patient profile dataclass for personalized diagnosis.
"""

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class PatientProfile:
    """Patient information for personalized diagnosis."""
    age: Optional[int] = None
    gender: Optional[str] = None
    weight: Optional[float] = None

    conditions: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)

    smoker: Optional[bool] = None
    alcohol_use: Optional[str] = None
    past_surgeries: List[str] = field(default_factory=list)

    # Pharmacogenomics (optional)
    cyp2d6: Optional[str] = None
    cyp2c19: Optional[str] = None
    hla_b5701: Optional[bool] = None

    def to_string(self) -> str:
        """Convert to readable string for LLM context."""
        parts = []

        if self.age:
            parts.append(f"**Age:** {self.age}")
        if self.gender:
            parts.append(f"**Gender:** {self.gender}")
        if self.weight:
            parts.append(f"**Weight:** {self.weight} kg")

        if self.conditions:
            parts.append(f"**Existing Conditions:** {', '.join(self.conditions)}")
        if self.allergies:
            parts.append(f"**⚠️ ALLERGIES:** {', '.join(self.allergies)}")
        if self.medications:
            parts.append(f"**Current Medications:** {', '.join(self.medications)}")
        if self.family_history:
            parts.append(f"**Family History:** {', '.join(self.family_history)}")
        if self.smoker is not None:
            parts.append(f"**Smoker:** {'Yes' if self.smoker else 'No'}")
        if self.alcohol_use:
            parts.append(f"**Alcohol:** {self.alcohol_use}")
        if self.past_surgeries:
            parts.append(f"**Past Surgeries:** {', '.join(self.past_surgeries)}")

        # Pharmacogenomics
        pgx = []
        if self.cyp2d6:
            pgx.append(f"CYP2D6: {self.cyp2d6}")
        if self.cyp2c19:
            pgx.append(f"CYP2C19: {self.cyp2c19}")
        if self.hla_b5701 is not None:
            pgx.append(f"HLA-B*57:01: {'Positive' if self.hla_b5701 else 'Negative'}")
        if pgx:
            parts.append(f"**Pharmacogenomics:** {', '.join(pgx)}")

        return "\n".join(parts) if parts else "No patient history provided"
