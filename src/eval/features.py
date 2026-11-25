"""Shared CPU-friendly feature extraction for offline evaluation and online gateway."""

from __future__ import annotations

import argparse
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

QUERY_META_COLUMNS = [
    "char_len",
    "token_count",
    "sentence_count",
    "question_mark_count",
    "digit_ratio",
    "uppercase_ratio",
    "avg_word_len",
]

RAG_FEATURE_COLUMNS = [
    "cosine_sim",
    "nli_entailment",
    "nli_contradiction",
    "nli_neutral",
    "entity_overlap",
    "length_ratio",
    "response_token_count",
    "context_token_count",
]

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_ENTITY_TOKEN_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]*|\d+(?:[.,]\d+)*)\b")
_DEFAULT_NLI_INDEX = {"contradiction": 0, "entailment": 1, "neutral": 2}


@dataclass
class FeatureExtractorConfig:
    """Configuration for embedding and NLI models used in feature extraction."""

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    embed_batch_size: int = 64
    nli_batch_size: int = 32
    nli_max_length: int = 512


def _clean_texts(texts: list[str]) -> list[str]:
    return ["" if t is None else str(t) for t in texts]


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _resolve_nli_label_indices(model: Any) -> dict[str, int]:
    """Read id2label dynamically from a CrossEncoder or model config with safe fallback."""
    id2label = getattr(getattr(getattr(model, "model", None), "config", None), "id2label", None)
    if id2label is None:
        id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not isinstance(id2label, dict) or not id2label:
        return dict(_DEFAULT_NLI_INDEX)

    resolved: dict[str, int] = {}
    for idx, label in id2label.items():
        name = str(label).lower()
        for target in ("entailment", "contradiction", "neutral"):
            if target in name:
                resolved[target] = int(idx)
    if len(resolved) == 3:
        return resolved
    return dict(_DEFAULT_NLI_INDEX)


def _extract_entities(text: str) -> set[str]:
    return {tok.lower() for tok in _ENTITY_TOKEN_RE.findall(text)}


def _call_with_supported_kwargs(fn: Any, arg: Any, kwargs: dict[str, Any]) -> Any:
    filtered = dict(kwargs)
    try:
        params = inspect.signature(fn).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            filtered = {k: v for k, v in filtered.items() if k in params}
    except (TypeError, ValueError):
        pass

    while True:
        try:
            return fn(arg, **filtered)
        except TypeError as exc:
            msg = str(exc)
            dropped = [k for k in filtered if f"'{k}'" in msg or f'"{k}"' in msg]
            if not dropped:
                return fn(arg)
            for k in dropped:
                filtered.pop(k, None)


def _compute_query_meta(queries: list[str]) -> pd.DataFrame:
    if not queries:
        return pd.DataFrame(
            {
                "char_len": pd.Series(dtype="int64"),
                "token_count": pd.Series(dtype="int64"),
                "sentence_count": pd.Series(dtype="int64"),
                "question_mark_count": pd.Series(dtype="int64"),
                "digit_ratio": pd.Series(dtype="float64"),
                "uppercase_ratio": pd.Series(dtype="float64"),
                "avg_word_len": pd.Series(dtype="float64"),
            }
        )

    rows = []
    for q in queries:
        char_len = len(q)
        words = q.split()
        token_count = len(words)
        if not q.strip():
            sentence_count = 0
        else:
            segs = [s for s in _SENTENCE_SPLIT_RE.split(q) if s.strip()]
            sentence_count = max(1, len(segs))
        question_mark_count = q.count("?")
        digit_ratio = sum(c.isdigit() for c in q) / char_len if char_len > 0 else 0.0
        uppercase_ratio = sum(c.isupper() for c in q) / char_len if char_len > 0 else 0.0
        avg_word_len = sum(len(w) for w in words) / token_count if token_count > 0 else 0.0
        rows.append(
            (
                char_len,
                token_count,
                sentence_count,
                question_mark_count,
                digit_ratio,
                uppercase_ratio,
                avg_word_len,
            )
        )
    return pd.DataFrame(rows, columns=QUERY_META_COLUMNS)


class FeatureExtractor:
    """Shared CPU-friendly extractor for query complexity, embeddings, and RAG NLI signals."""

    def __init__(
        self,
        config: FeatureExtractorConfig | None = None,
        embedder: Any | None = None,
        nli_model: Any | None = None,
    ) -> None:
        self.config = config or FeatureExtractorConfig()
        self._embedder = embedder
        self._nli_model = nli_model

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.config.embed_model)
        return self._embedder

    def _get_nli_model(self) -> Any:
        if self._nli_model is None:
            from sentence_transformers import CrossEncoder

            self._nli_model = CrossEncoder(
                self.config.nli_model,
                max_length=self.config.nli_max_length,
            )
        return self._nli_model

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = 384
            if self._embedder is not None and hasattr(self._embedder, "get_sentence_embedding_dimension"):
                maybe_dim = self._embedder.get_sentence_embedding_dimension()
                if maybe_dim is not None:
                    dim = int(maybe_dim)
            return np.empty((0, dim), dtype=np.float32)

        embedder = self._get_embedder()
        unique_texts = list(dict.fromkeys(texts))
        encoded = _call_with_supported_kwargs(
            embedder.encode,
            unique_texts,
            {
                "batch_size": self.config.embed_batch_size,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            },
        )

        unique_embs = _l2_normalize(np.asarray(encoded, dtype=np.float32))
        idx_map = {txt: i for i, txt in enumerate(unique_texts)}
        indices = [idx_map[txt] for txt in texts]
        return unique_embs[indices]

    def _predict_nli(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not pairs:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty

        model = self._get_nli_model()
        label_idx = _resolve_nli_label_indices(model)
        raw = _call_with_supported_kwargs(
            model.predict,
            pairs,
            {
                "batch_size": self.config.nli_batch_size,
                "apply_softmax": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            },
        )

        scores = np.asarray(raw, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        if (scores < 0).any() or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
            scores = _softmax(scores)

        return (
            scores[:, label_idx["entailment"]],
            scores[:, label_idx["contradiction"]],
            scores[:, label_idx["neutral"]],
        )

    def extract_query_features(self, queries: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute L2-normalized MiniLM query embeddings and lexical complexity metadata."""
        clean_queries = _clean_texts(queries)
        embs = self._encode_texts(clean_queries)
        meta = _compute_query_meta(clean_queries)
        return embs, meta

    def extract_rag_features(
        self,
        queries: list[str],
        contexts: list[str],
        responses: list[str],
    ) -> pd.DataFrame:
        """Compute retrieval-to-answer cosine similarity, NLI scores, and overlap features."""
        if not (len(queries) == len(contexts) == len(responses)):
            raise ValueError(
                "queries, contexts, and responses must have the same length: "
                f"got {len(queries)}, {len(contexts)}, {len(responses)}"
            )
        if not queries:
            return pd.DataFrame(
                {
                    "cosine_sim": pd.Series(dtype="float64"),
                    "nli_entailment": pd.Series(dtype="float64"),
                    "nli_contradiction": pd.Series(dtype="float64"),
                    "nli_neutral": pd.Series(dtype="float64"),
                    "entity_overlap": pd.Series(dtype="float64"),
                    "length_ratio": pd.Series(dtype="float64"),
                    "response_token_count": pd.Series(dtype="int64"),
                    "context_token_count": pd.Series(dtype="int64"),
                }
            )

        clean_ctx = _clean_texts(contexts)
        clean_resp = _clean_texts(responses)

        ctx_embs = self._encode_texts(clean_ctx)
        resp_embs = self._encode_texts(clean_resp)
        cosine_sim = np.clip(np.sum(ctx_embs * resp_embs, axis=1), -1.0, 1.0).astype(np.float64)
        for i, (ctx, resp) in enumerate(zip(clean_ctx, clean_resp)):
            if not ctx.strip() or not resp.strip():
                cosine_sim[i] = 0.0

        nli_entail, nli_contra, nli_neutral = self._predict_nli(list(zip(clean_ctx, clean_resp)))

        entity_overlaps = []
        length_ratios = []
        resp_tok_counts = []
        ctx_tok_counts = []
        for ctx, resp in zip(clean_ctx, clean_resp):
            ctx_toks = len(ctx.split())
            resp_toks = len(resp.split())
            ctx_tok_counts.append(ctx_toks)
            resp_tok_counts.append(resp_toks)
            length_ratios.append(resp_toks / ctx_toks if ctx_toks > 0 else 0.0)

            ctx_ents = _extract_entities(ctx)
            resp_ents = _extract_entities(resp)
            if not ctx_ents or not resp_ents:
                entity_overlaps.append(0.0)
            else:
                entity_overlaps.append(len(ctx_ents & resp_ents) / len(resp_ents))

        return pd.DataFrame(
            {
                "cosine_sim": cosine_sim,
                "nli_entailment": nli_entail,
                "nli_contradiction": nli_contra,
                "nli_neutral": nli_neutral,
                "entity_overlap": np.asarray(entity_overlaps, dtype=np.float64),
                "length_ratio": np.asarray(length_ratios, dtype=np.float64),
                "response_token_count": np.asarray(resp_tok_counts, dtype="int64"),
                "context_token_count": np.asarray(ctx_tok_counts, dtype="int64"),
            },
            columns=RAG_FEATURE_COLUMNS,
        )


def _get_query_column(df: pd.DataFrame, dataset_name: str) -> str:
    for col in ("prompt", "query"):
        if col in df.columns:
            return col
    raise ValueError(f"{dataset_name} missing required query column ('prompt' or 'query')")


def build_feature_tables(
    ragtruth_path: Path,
    routerbench_path: Path,
    out_dir: Path,
    extractor: FeatureExtractor | None = None,
) -> tuple[Path, Path]:
    """Extract and persist feature Parquet tables for RAGTruth and RouterBench."""
    ext = extractor or FeatureExtractor()
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rag_df = pd.read_parquet(ragtruth_path).reset_index(drop=True)
    rag_q_col = _get_query_column(rag_df, "RAGTruth")
    missing_rag = [col for col in ("context", "response") if col not in rag_df.columns]
    if missing_rag:
        raise ValueError(f"RAGTruth missing required columns: {missing_rag}")
    rag_queries = rag_df[rag_q_col].fillna("").astype(str).tolist()
    rag_contexts = rag_df["context"].fillna("").astype(str).tolist()
    rag_responses = rag_df["response"].fillna("").astype(str).tolist()

    rag_q_meta = _compute_query_meta(rag_queries)
    rag_feats = ext.extract_rag_features(rag_queries, rag_contexts, rag_responses)
    rag_out_df = pd.concat([rag_df, rag_q_meta, rag_feats], axis=1)
    rag_out_path = out_root / "ragtruth_features.parquet"
    rag_out_df.to_parquet(rag_out_path, index=False)

    rtr_df = pd.read_parquet(routerbench_path).reset_index(drop=True)
    rtr_q_col = _get_query_column(rtr_df, "RouterBench")
    rtr_queries = rtr_df[rtr_q_col].fillna("").astype(str).tolist()

    rtr_embs, rtr_q_meta = ext.extract_query_features(rtr_queries)
    rtr_out_df = pd.concat([rtr_df, rtr_q_meta], axis=1)
    rtr_table = pa.Table.from_pandas(rtr_out_df, preserve_index=False)
    emb_array = pa.array([row.tolist() for row in rtr_embs.astype(np.float32)], type=pa.list_(pa.float32()))
    rtr_table = rtr_table.append_column("query_embedding", emb_array)

    rtr_out_path = out_root / "routerbench_features.parquet"
    pq.write_table(rtr_table, rtr_out_path)

    return rag_out_path, rtr_out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Parquet feature tables for RAGTruth and RouterBench.")
    parser.add_argument("--config", type=Path, default=Path("config/ragtruth_conformal.yaml"))
    parser.add_argument("--ragtruth-in", type=Path, default=Path("data/raw/ragtruth.parquet"))
    parser.add_argument("--routerbench-in", type=Path, default=Path("data/raw/routerbench.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    cfg = FeatureExtractorConfig()
    if args.config.exists():
        raw_cfg = yaml.safe_load(args.config.read_text()) or {}
        if raw_cfg.get("embed_model") is not None:
            cfg.embed_model = str(raw_cfg["embed_model"])
        if raw_cfg.get("nli_model") is not None:
            cfg.nli_model = str(raw_cfg["nli_model"])

    rag_out, rtr_out = build_feature_tables(
        ragtruth_path=args.ragtruth_in,
        routerbench_path=args.routerbench_in,
        out_dir=args.out_dir,
        extractor=FeatureExtractor(config=cfg),
    )
    rag_meta = pq.read_metadata(rag_out)
    rtr_meta = pq.read_metadata(rtr_out)
    print(f"Saved RAGTruth features -> {rag_out} ({rag_meta.num_rows} rows, {rag_meta.num_columns} cols)")
    print(f"Saved RouterBench features -> {rtr_out} ({rtr_meta.num_rows} rows, {rtr_meta.num_columns} cols)")


if __name__ == "__main__":
    main()
