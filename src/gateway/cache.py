"""FAISS-backed semantic cache with conformal abstention filtering."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
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

    def lookup(
        self,
        embedding: np.ndarray,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any] | None:
        """Return first cached SQLite record >= threshold matching predicate, else None."""
        vec = self._normalize(embedding)
        if vec is None:
            return None

        with self._lock:
            ntotal = int(self._index.ntotal)
            if ntotal == 0:
                return None
            k = min(16, ntotal)
            scores, indices = self._index.search(vec, k)

        for score_raw, idx_raw in zip(scores[0], indices[0]):
            score = float(score_raw)
            faiss_id = int(idx_raw)
            if faiss_id < 0 or score < self.similarity_threshold:
                continue
            record = self.store.get_cached_request_by_faiss_id(faiss_id)
            if record is None or record.get("conformal_decision") == "ABSTAIN":
                continue
            if predicate is None or predicate(record):
                return record
        return None

    def put(self, embedding: np.ndarray, record: dict[str, Any]) -> None:
        """Persist non-ABSTAIN request embedding to FAISS atomically and register in SQLite."""
        vec = self._normalize(embedding)
        if record.get("conformal_decision") == "ABSTAIN" or vec is None:
            return

        request_id = record["request_id"]
        tmp_path = self.index_path.with_suffix(".tmp")
        with self._lock:
            if self.store.get_request(request_id) is None:
                self.store.insert_request(record)
            faiss_id = int(self._index.ntotal)
            self._index.add(vec)
            try:
                faiss.write_index(self._index, str(tmp_path))
                self.store.register_cache_entry(faiss_id, request_id)
                os.replace(tmp_path, self.index_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                self._index.remove_ids(np.array([faiss_id], dtype=np.int64))
                raise
