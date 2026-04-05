import os
import pickle
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(self.embedding_model)

    def build_from_documents(self, documents) -> None:
        print("[INFO] Building vector store from documents...")

        # Reset current in-memory state before rebuilding
        self.index = None
        self.metadata = []

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        metadatas: List[Dict[str, Any]] = []
        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            meta["text"] = chunk.page_content
            metadatas.append(meta)

        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]] | None = None) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D numpy array.")

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Cannot save because the FAISS index has not been built yet.")

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self) -> None:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"FAISS index files not found in '{self.persist_dir}'. Build the index first."
            )

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def exists(self) -> bool:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        return os.path.exists(faiss_path) and os.path.exists(meta_path)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Vector index is not loaded. Build or load the store first.")

        distances, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append(
                {
                    "index": int(idx),
                    "distance": float(dist),
                    "metadata": meta,
                }
            )

        return results

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)


if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))