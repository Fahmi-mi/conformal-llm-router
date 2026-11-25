"""FAISS-backed semantic cache with conformal abstention filtering."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.gateway.storage import TelemetryStore


class SemanticCache:
    """Cosine-similarity semantic cache backed by FAISS IndexFlatIP and SQLite."""

    def __init__(
        self,
        index_path: Path,
        store: TelemetryStore,
        similarity_threshold: float = 0.92,
        embedding_dim: int = 384,
    ) -> None:
        self.index_path = Path(index_path)
        self.store = store
        self.similarity_threshold = float(similarity_threshold)
        self.embedding_dim = int(embedding_dim)
        self._lock = threading.Lock()

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            if self._index.d != self.embedding_dim:
                raise ValueError(
                    f"Loaded FAISS index dimension {self._index.d} does not match "
                    f"expected embedding_dim {self.embedding_dim}."
                )
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)

    def _normalize(self, embedding: np.ndarray) -> np.ndarray | None:
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2 or arr.shape[0] != 1:
            raise ValueError(
                f"Expected 1D or (1, {self.embedding_dim}) embedding, got shape {arr.shape}."
            )
        if arr.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {arr.shape[1]}."
            )
        if not np.all(np.isfinite(arr)):
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            return None
        return np.ascontiguousarray(arr / norm, dtype=np.float32)

    def lookup(self, embedding: np.ndarray) -> dict[str, Any] | None:
        """Return cached SQLite record if top cosine similarity >= threshold, else None."""
        vec = self._normalize(embedding)
        if vec is None:
            return None

        with self._lock:
            if self._index.ntotal == 0:
                return None
            scores, indices = self._index.search(vec, 1)

        top_score = float(scores[0][0])
        top_id = int(indices[0][0])
        if top_id < 0 or top_score < self.similarity_threshold:
            return None

        record = self.store.get_cached_request_by_faiss_id(top_id)
        if record is None or record.get("conformal_decision") == "ABSTAIN":
            return None
        return record

    def put(self, embedding: np.ndarray, record: dict[str, Any]) -> None:
        """Persist non-ABSTAIN request embedding to FAISS and register in SQLite."""
        vec = self._normalize(embedding)
        if record.get("conformal_decision") == "ABSTAIN" or vec is None:
            return

        request_id = record["request_id"]
        with self._lock:
            if self.store.get_request(request_id) is None:
                self.store.insert_request(record)
            faiss_id = int(self._index.ntotal)
            self._index.add(vec)
            faiss.write_index(self._index, str(self.index_path))
            self.store.register_cache_entry(faiss_id, request_id)
