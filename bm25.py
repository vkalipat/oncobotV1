"""
Lightweight BM25 implementation for hybrid retrieval.
"""

import re
import math
from typing import List, Any, Tuple
from collections import Counter


class BM25:
    """Lightweight BM25 implementation for hybrid retrieval."""

    def __init__(self, documents: List[Any], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)

        # Tokenize
        self.doc_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avg_dl = sum(self.doc_lens) / max(self.doc_count, 1)

        # Build IDF
        self.idf = {}
        df = Counter()
        for tokens in self.doc_tokens:
            unique = set(tokens)
            for token in unique:
                df[token] += 1

        for token, freq in df.items():
            self.idf[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

    def _tokenize(self, text: str) -> List[str]:
        """Simple medical-aware tokenization."""
        text = text.lower()
        # Keep medical terms with hyphens and numbers
        tokens = re.findall(r'[a-z0-9][\w\-]*[a-z0-9]|[a-z0-9]', text)
        return tokens

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by relevance."""
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc_tokens in enumerate(self.doc_tokens):
            score = 0.0
            dl = self.doc_lens[idx]
            tf_map = Counter(doc_tokens)

            for qt in query_tokens:
                if qt in self.idf:
                    tf = tf_map.get(qt, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                    score += self.idf[qt] * numerator / denominator

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
