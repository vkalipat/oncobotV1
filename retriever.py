"""
Hybrid document retriever - FAISS + BM25 + cross-encoder re-ranking.
"""

import os
import re
import hashlib
from typing import List, Any

from langchain_community.vectorstores import FAISS

from .config import TOP_K_INITIAL, TOP_K_RERANKED
from .bm25 import BM25
from .reranker import CrossEncoderReranker
from .query_expander import QueryExpander


class DocumentRetriever:
    """Hybrid document retrieval using FAISS + BM25 + cross-encoder re-ranking."""

    def __init__(self, vectorstore: FAISS, use_cross_encoder: bool = True):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_INITIAL}
        )
        self.query_expander = QueryExpander()

        # Initialize BM25 for hybrid retrieval
        try:
            all_docs_dict = self.vectorstore.docstore._dict
            self.all_docs = list(all_docs_dict.values())
            self.bm25 = BM25(self.all_docs)
            print(f"✓ BM25 index built with {len(self.all_docs)} chunks")
        except Exception as e:
            self.bm25 = None
            self.all_docs = []
            print(f"⚠ BM25 not available: {e}")

        # Cross-encoder re-ranker
        if use_cross_encoder:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None

    def retrieve_context(self, query: str) -> str:
        """Hybrid retrieval with FAISS + BM25 + cross-encoder re-ranking."""

        # 1. Multi-query expansion
        queries = self.query_expander.expand_medical_queries(query)

        # 2. FAISS retrieval (semantic)
        all_faiss_docs = []
        seen_contents = set()
        for q in queries:
            try:
                docs = self.retriever.invoke(q)
                for doc in docs:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_faiss_docs.append(doc)
            except:
                pass

        # 3. BM25 retrieval (lexical)
        bm25_docs = []
        if self.bm25:
            for q in queries[:2]:  # Use fewer queries for BM25
                results = self.bm25.search(q, top_k=8)
                for idx, _ in results:
                    doc = self.all_docs[idx]
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        bm25_docs.append(doc)

        # 4. Merge candidates
        candidates = all_faiss_docs + bm25_docs

        if not candidates:
            return "No relevant documents found in medical reference library."

        # 5. Cross-encoder re-ranking
        if self.reranker and self.reranker.model:
            final_docs = self.reranker.rerank(query, candidates, top_k=TOP_K_RERANKED)
        else:
            # Fallback: keyword-based scoring
            final_docs = self._keyword_rerank(query, candidates, top_k=TOP_K_RERANKED)

        # 6. Format context
        chunks = []
        for i, doc in enumerate(final_docs, 1):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', '?')
            content = doc.page_content[:800]
            chunks.append(f"[Source: {source}, Page {page}]\n{content}")

        return "\n\n---\n\n".join(chunks)

    def _keyword_rerank(self, query: str, documents: List[Any], top_k: int = 6) -> List[Any]:
        """Fallback keyword-based re-ranking when cross-encoder is not available."""
        query_terms = set(re.findall(r'\w+', query.lower()))

        scored = []
        for doc in documents:
            doc_terms = set(re.findall(r'\w+', doc.page_content.lower()))
            overlap = len(query_terms & doc_terms)
            scored.append((doc, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]
