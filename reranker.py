"""
Cross-encoder re-ranker for document relevance scoring.
"""

from typing import List, Any

from .config import CROSS_ENCODER_MODEL, CROSS_ENCODER_AVAILABLE, TOP_K_RERANKED

if CROSS_ENCODER_AVAILABLE:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Re-rank retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        if not CROSS_ENCODER_AVAILABLE:
            self.model = None
            print("⚠ Cross-encoder not available. Install: pip install sentence-transformers")
            return

        try:
            self.model = CrossEncoder(model_name, max_length=512)
            print(f"✓ Cross-encoder loaded: {model_name}")
        except Exception as e:
            self.model = None
            print(f"⚠ Could not load cross-encoder: {e}")

    def rerank(self, query: str, documents: List[Any], top_k: int = TOP_K_RERANKED) -> List[Any]:
        """Re-rank documents by cross-encoder relevance score."""
        if not self.model or not documents:
            return documents[:top_k]

        # Create query-document pairs
        pairs = [(query, doc.page_content[:500]) for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]
