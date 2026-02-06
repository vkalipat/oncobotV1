"""
Configuration constants, enums, and model recommendations.
"""

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

# Provider-specific imports
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False


# =============================================================================
# PATHS & CHUNKING
# =============================================================================

DOCS_PATH = "docs"
VECTORSTORE_PATH = "faiss_index_medical"

# Optimized for medical documents - larger chunks preserve clinical context
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Best general-purpose embedding for retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval settings
TOP_K_INITIAL = 15       # Pull more candidates for re-ranking
TOP_K_RERANKED = 6       # Final context chunks after re-ranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# BM25 settings
BM25_K1 = 1.5
BM25_B = 0.75


# =============================================================================
# ENUMS
# =============================================================================

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# =============================================================================
# MODEL RECOMMENDATIONS (2025)
# =============================================================================

MODEL_RECOMMENDATIONS = {
    "best_accuracy": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-4o",
        "description": "Highest USMLE benchmark scores (~92%), best for complex differential diagnosis"
    },
    "best_reasoning": {
        "provider": LLMProvider.ANTHROPIC,
        "model": "claude-sonnet-4-20250514",
        "description": "Best clinical reasoning and explanation, safety-conscious, excellent at treatment personalization"
    },
    "best_value": {
        "provider": LLMProvider.ANTHROPIC,
        "model": "claude-sonnet-4-20250514",
        "description": "Claude Sonnet 4 - top-tier accuracy at moderate cost, best reasoning chain"
    },
    "fastest_cheapest": {
        "provider": LLMProvider.GOOGLE,
        "model": "gemini-2.0-flash",
        "description": "Fast and cheap, good for triage and simple cases"
    },
    "budget_accurate": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-4o-mini",
        "description": "Good balance of cost and accuracy for routine cases"
    }
}
