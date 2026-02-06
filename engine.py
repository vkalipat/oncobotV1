"""
DiagnosticEngine - orchestrates retrieval, web search, and clinical pipeline.
"""

from typing import Dict, List, Optional, Any

from langchain_community.vectorstores import FAISS

from .config import LLMProvider, WEB_SEARCH_AVAILABLE
from .patient import PatientProfile
from .vectorstore import VectorStoreManager
from .retriever import DocumentRetriever
from .web_search import WebSearcher
from .llm_factory import LLMFactory
from .clinical_pipeline import ClinicalPipeline


class DiagnosticEngine:
    """
    Production-grade multi-provider diagnostic engine.

    Orchestrates all components: retrieval, web search, clinical pipeline.

    Features:
    - Hybrid retrieval (FAISS + BM25)
    - Cross-encoder re-ranking
    - Multi-query expansion
    - 5-stage clinical pipeline
    - Pharmacogenomics-aware

    Usage:
        engine = DiagnosticEngine(provider="anthropic", model="claude-sonnet-4-20250514")
        result = engine.diagnose(symptoms="...", session_id="patient123")
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = None,
        api_key: str = None,
        vectorstore: FAISS = None,
        use_web_search: bool = True,
        use_cross_encoder: bool = True,
        use_multi_stage: bool = True
    ):
        # Parse provider
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())

        self.provider = provider
        self.model = model
        self.use_web_search = use_web_search
        self.use_multi_stage = use_multi_stage

        # Create LLM
        self.llm = LLMFactory.create_llm(provider, model, api_key)

        # Load vectorstore
        if vectorstore is None:
            vs_manager = VectorStoreManager()
            vectorstore = vs_manager.load_or_create_vectorstore()
        self.vectorstore = vectorstore

        # Initialize retriever
        self.document_retriever = DocumentRetriever(vectorstore, use_cross_encoder)

        # Initialize web searcher
        self.web_searcher = WebSearcher()

        # Initialize clinical pipeline
        self.clinical_pipeline = ClinicalPipeline(self.llm)

        # Patient sessions
        self.patient_sessions: Dict[str, PatientProfile] = {}

        print(f"✓ Initialized with {provider.value} ({model or 'default model'})")
        reranker = self.document_retriever.reranker
        print(f"  Retrieval: FAISS + {'BM25' if self.document_retriever.bm25 else 'no BM25'} | Re-ranking: {'Cross-encoder' if reranker and reranker.model else 'Keyword'}")

    def get_patient(self, session_id: str) -> PatientProfile:
        if session_id not in self.patient_sessions:
            self.patient_sessions[session_id] = PatientProfile()
        return self.patient_sessions[session_id]

    def set_patient(self, session_id: str, profile: PatientProfile):
        self.patient_sessions[session_id] = profile

    def retrieve_context(self, query: str) -> str:
        """Hybrid retrieval with FAISS + BM25 + cross-encoder re-ranking."""
        return self.document_retriever.retrieve_context(query)

    def _keyword_rerank(self, query: str, documents: List[Any], top_k: int = 6) -> List[Any]:
        """Fallback keyword-based re-ranking when cross-encoder is not available."""
        return self.document_retriever._keyword_rerank(query, documents, top_k)

    def diagnose(
        self,
        symptoms: str,
        session_id: str = "default",
        patient_info: Dict = None,
        use_web_search: bool = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diagnosis using multi-stage clinical pipeline.

        Args:
            symptoms: Patient symptom description
            session_id: Session ID for patient tracking
            patient_info: Optional dict to update patient profile
            use_web_search: Override web search setting

        Returns:
            Dict with diagnosis, sources, pipeline stages, metadata
        """
        # Get/update patient
        patient = self.get_patient(session_id)

        if patient_info:
            for key, value in patient_info.items():
                if hasattr(patient, key):
                    if isinstance(getattr(patient, key), list) and isinstance(value, str):
                        setattr(patient, key, [v.strip() for v in value.split(",") if v.strip()])
                    else:
                        setattr(patient, key, value)

        patient_str = patient.to_string()

        # Retrieve context
        doc_context = self.retrieve_context(symptoms)

        # Web search if enabled and needed
        web_context = ""
        web_used = False
        should_search = use_web_search if use_web_search is not None else self.use_web_search
        if should_search and WEB_SEARCH_AVAILABLE:
            if len(doc_context) < 300 or "No relevant" in doc_context:
                web_context = self.web_searcher.search_medical_literature(symptoms)
                web_used = bool(web_context)

        # Multi-stage pipeline or single-shot
        if self.use_multi_stage:
            diagnosis = self.clinical_pipeline.multi_stage_diagnose(
                symptoms=symptoms,
                patient_str=patient_str,
                doc_context=doc_context,
                web_context=web_context
            )
        else:
            diagnosis = self.clinical_pipeline.single_shot_diagnose(
                symptoms=symptoms,
                patient_str=patient_str,
                doc_context=doc_context,
                web_context=web_context
            )

        reranker = self.document_retriever.reranker
        return {
            "diagnosis": diagnosis,
            "symptoms": symptoms,
            "patient_info": patient_str,
            "document_context_used": len(doc_context) > 100 and "No relevant" not in doc_context,
            "web_search_used": web_used,
            "provider": self.provider.value,
            "model": self.model or "default",
            "pipeline": "multi-stage" if self.use_multi_stage else "single-shot",
            "retrieval": f"FAISS+BM25+{'CrossEncoder' if reranker and reranker.model else 'Keyword'}Rerank"
        }

    def _multi_stage_diagnose(self, symptoms: str, patient_str: str, doc_context: str, web_context: str) -> str:
        """5-stage clinical pipeline for maximum accuracy."""
        return self.clinical_pipeline.multi_stage_diagnose(symptoms, patient_str, doc_context, web_context)

    def _single_shot_diagnose(self, symptoms: str, patient_str: str, doc_context: str, web_context: str) -> str:
        """Single-shot diagnosis for faster response (fallback mode)."""
        return self.clinical_pipeline.single_shot_diagnose(symptoms, patient_str, doc_context, web_context)

    def quick_diagnose(
        self,
        symptoms: str,
        age: int = None,
        gender: str = None,
        conditions: List[str] = None,
        allergies: List[str] = None,
        medications: List[str] = None
    ) -> str:
        """Quick one-shot diagnosis. Returns just the diagnosis text."""
        patient = PatientProfile(
            age=age,
            gender=gender,
            conditions=conditions or [],
            allergies=allergies or [],
            medications=medications or []
        )

        doc_context = self.retrieve_context(symptoms)
        web_context = self.web_searcher.search_medical_literature(symptoms) if self.use_web_search else ""

        return self.clinical_pipeline.single_shot_diagnose(
            symptoms=symptoms,
            patient_str=patient.to_string(),
            doc_context=doc_context,
            web_context=web_context
        )
