"""
Diagnostic RAG Pipeline package.

All classes and functions are available from this top-level import:
    from pipeline import DiagnosticEngine, PatientProfile, load_or_create_vectorstore
"""

from .config import (
    DOCS_PATH,
    VECTORSTORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    TOP_K_INITIAL,
    TOP_K_RERANKED,
    CROSS_ENCODER_MODEL,
    BM25_K1,
    BM25_B,
    LLMProvider,
    MODEL_RECOMMENDATIONS,
    OPENAI_AVAILABLE,
    ANTHROPIC_AVAILABLE,
    GOOGLE_AVAILABLE,
    CROSS_ENCODER_AVAILABLE,
    WEB_SEARCH_AVAILABLE,
)
from .patient import PatientProfile
from .prompts import (
    TRIAGE_PROMPT,
    DIFFERENTIAL_PROMPT,
    WORKUP_PROMPT,
    TREATMENT_PROMPT,
    ASSEMBLY_PROMPT,
    TRIAGE_DIFFERENTIAL_PROMPT,
    WORKUP_TREATMENT_PROMPT,
)
from .bm25 import BM25
from .reranker import CrossEncoderReranker
from .vectorstore import VectorStoreManager
from .web_search import WebSearcher
from .query_expander import QueryExpander
from .llm_factory import LLMFactory
from .retriever import DocumentRetriever
from .clinical_pipeline import ClinicalPipeline
from .engine import DiagnosticEngine


# Backward-compatible free functions

def load_or_create_vectorstore(
    docs_path: str = DOCS_PATH,
    vectorstore_path: str = VECTORSTORE_PATH,
):
    """Load existing vectorstore or create from documents."""
    manager = VectorStoreManager(docs_path=docs_path, vectorstore_path=vectorstore_path)
    return manager.load_or_create_vectorstore()


def search_medical_literature(query: str, num_results: int = 5) -> str:
    """Search for additional medical information via web."""
    return WebSearcher().search_medical_literature(query, num_results)


def expand_medical_queries(symptoms: str):
    """Generate multiple search queries from symptoms for better recall."""
    return QueryExpander().expand_medical_queries(symptoms)


def create_llm(provider, model=None, api_key=None, temperature=0.1):
    """Create LLM instance based on provider with best available models."""
    return LLMFactory.create_llm(provider, model, api_key, temperature)


def setup_diagnostic_engine(provider="anthropic", model=None, api_key=None):
    """Setup diagnostic engine with specified provider."""
    return DiagnosticEngine(provider=provider, model=model, api_key=api_key)
