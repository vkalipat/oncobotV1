"""
VectorStore manager - creation, loading, and cache invalidation.

The embeddings model and vectorstore are cached in memory so documents are
ingested/chunked exactly once per process, then reused for every query.
"""

import os
import hashlib

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    DOCS_PATH,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# Module-level caches — shared across all VectorStoreManager instances
_embeddings_cache: dict[str, HuggingFaceEmbeddings] = {}
_vectorstore_cache: dict[str, FAISS] = {}


class VectorStoreManager:
    """Handles vectorstore creation, loading, and cache invalidation.

    The embedding model is loaded once per model name and the vectorstore is
    loaded/built once per (docs_path, vectorstore_path) pair.  Subsequent
    calls return the cached objects instantly.
    """

    def __init__(
        self,
        docs_path: str = DOCS_PATH,
        vectorstore_path: str = VECTORSTORE_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.docs_path = docs_path
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Reuse the embedding model if it was already loaded
        if embedding_model not in _embeddings_cache:
            _embeddings_cache[embedding_model] = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        self.embeddings = _embeddings_cache[embedding_model]

    def _compute_docs_hash(self) -> str:
        """Compute hash of all documents for cache invalidation."""
        if not os.path.exists(self.docs_path):
            return ""

        h = hashlib.md5()
        for fname in sorted(os.listdir(self.docs_path)):
            fpath = os.path.join(self.docs_path, fname)
            if os.path.isfile(fpath):
                h.update(fname.encode())
                h.update(str(os.path.getmtime(fpath)).encode())
                h.update(str(os.path.getsize(fpath)).encode())
        return h.hexdigest()

    def load_or_create_vectorstore(self) -> FAISS:
        """Load existing vectorstore or create from documents.

        Returns a cached instance if the documents haven't changed since
        the last call.
        """
        cache_key = f"{self.docs_path}::{self.vectorstore_path}"
        docs_hash = self._compute_docs_hash()

        # Return in-memory cache if docs are unchanged
        if cache_key in _vectorstore_cache:
            hash_file = os.path.join(self.vectorstore_path, "docs_hash.txt")
            stored_hash = ""
            if os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    stored_hash = f.read().strip()
            if stored_hash == docs_hash:
                print(f"✓ Reusing in-memory vectorstore ({len(_vectorstore_cache[cache_key].docstore._dict)} chunks)")
                return _vectorstore_cache[cache_key]
            else:
                # Docs changed — drop the stale cache entry
                del _vectorstore_cache[cache_key]
                print("📄 Documents changed, rebuilding vectorstore...")

        # Try loading from disk
        hash_file = os.path.join(self.vectorstore_path, "docs_hash.txt")

        if os.path.exists(self.vectorstore_path):
            try:
                stored_hash = ""
                if os.path.exists(hash_file):
                    with open(hash_file, 'r') as f:
                        stored_hash = f.read().strip()

                if stored_hash == docs_hash:
                    vectorstore = FAISS.load_local(
                        self.vectorstore_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    _vectorstore_cache[cache_key] = vectorstore
                    print(f"✓ Loaded vectorstore from {self.vectorstore_path} (docs unchanged)")
                    return vectorstore
                else:
                    print("📄 Documents changed, rebuilding vectorstore...")
            except Exception as e:
                print(f"Could not load vectorstore: {e}")

        # Create new from PDFs
        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)
            raise FileNotFoundError(f"Created '{self.docs_path}'. Add medical PDFs and restart.")

        if not os.listdir(self.docs_path):
            raise FileNotFoundError(f"'{self.docs_path}' is empty. Add medical PDFs.")

        # Load documents
        loader = PyPDFDirectoryLoader(self.docs_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No documents loaded from PDFs.")

        # Medical-optimized splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n## ",      # Major section headers
                "\n### ",     # Sub-section headers
                "\n#### ",    # Sub-sub headers
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                ". ",         # Sentence boundaries
                " ",          # Word boundaries
            ]
        )
        splits = splitter.split_documents(documents)

        # Create vectorstore
        vectorstore = FAISS.from_documents(splits, self.embeddings)

        # Save with hash
        vectorstore.save_local(self.vectorstore_path)
        os.makedirs(self.vectorstore_path, exist_ok=True)
        with open(hash_file, 'w') as f:
            f.write(docs_hash)

        _vectorstore_cache[cache_key] = vectorstore
        print(f"✓ Created vectorstore with {len(splits)} chunks from {len(documents)} pages")
        return vectorstore
