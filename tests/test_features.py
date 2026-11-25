from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.features import (
    FeatureExtractor,
    FeatureExtractorConfig,
    build_feature_tables,
)


class FakeEmbedder:
    """Deterministic lightweight embedder stub returning 384-d vectors."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.seen_texts: list[str] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.seen_texts.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if not text.strip():
                out[i, 0] = 1.0
            else:
                seed = sum(ord(c) for c in text)
                out[i, seed % self.dim] = 1.0
                out[i, (seed + 7) % self.dim] = 0.5
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
        return out


class _FakeConfig:
    def __init__(self, id2label: dict[int, str]) -> None:
        self.id2label = id2label


class _FakeInnerModel:
    def __init__(self, id2label: dict[int, str]) -> None:
        self.config = _FakeConfig(id2label)


class FakeNLIModel:
    """Lightweight CrossEncoder stub with configurable id2label ordering."""

    def __init__(self, id2label: dict[int, str] | None = None) -> None:
        if id2label is None:
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.model = _FakeInnerModel(id2label)
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        apply_softmax: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.seen_pairs.extend(pairs)
        logits = np.tile(np.array([[0.2, 2.0, 0.5]], dtype=np.float32), (len(pairs), 1))
        if not apply_softmax:
            return logits
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


def test_extract_rag_features_handles_empty_strings_without_nan():
    ext = FeatureExtractor(embedder=FakeEmbedder(), nli_model=FakeNLIModel())
    df = ext.extract_rag_features(["q"], [""], [""])
    assert not df.isna().any().any()
    assert df["cosine_sim"].iloc[0] == 0.0
    assert df["length_ratio"].iloc[0] == 0.0
    assert df["entity_overlap"].iloc[0] == 0.0
    assert df["response_token_count"].iloc[0] == 0
    assert df["context_token_count"].iloc[0] == 0


def test_extract_query_features_shape_and_l2_norm():
    ext = FeatureExtractor(embedder=FakeEmbedder(), nli_model=FakeNLIModel())
    embs, meta = ext.extract_query_features(["Explain conformal prediction.", "2 + 2?"])
    assert embs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4)
    expected_cols = {
        "char_len",
        "token_count",
        "sentence_count",
        "question_mark_count",
        "digit_ratio",
        "uppercase_ratio",
        "avg_word_len",
    }
    assert expected_cols.issubset(meta.columns)
    assert meta["question_mark_count"].tolist() == [0, 1]
    assert meta["sentence_count"].tolist() == [1, 1]
    assert meta["digit_ratio"].iloc[1] > 0.0


def test_extract_rag_features_truncates_long_passages_safely():
    ext = FeatureExtractor(embedder=FakeEmbedder(), nli_model=FakeNLIModel())
    long_ctx = "word " * 2000
    df = ext.extract_rag_features(["q"], [long_ctx], ["short answer"])
    assert np.isfinite(df["cosine_sim"].iloc[0])
    assert not df.isna().any().any()


def test_extract_rag_features_dynamic_id2label_and_softmax_sum():
    nli = FakeNLIModel(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    ext = FeatureExtractor(embedder=FakeEmbedder(), nli_model=nli)
    df = ext.extract_rag_features(
        ["Where is Paris?"],
        ["Paris is the capital of France."],
        ["Paris is in France."],
    )
    prob_sum = (
        df["nli_entailment"].iloc[0]
        + df["nli_contradiction"].iloc[0]
        + df["nli_neutral"].iloc[0]
    )
    assert np.isclose(prob_sum, 1.0, atol=1e-5)
    # Row logits are [0.2, 2.0, 0.5] -> index 1 ("neutral") has highest probability
    assert df["nli_neutral"].iloc[0] > df["nli_entailment"].iloc[0]
    assert df["nli_neutral"].iloc[0] > df["nli_contradiction"].iloc[0]
    assert df["entity_overlap"].iloc[0] == 1.0


def test_extract_features_empty_inputs_and_mismatched_lengths():
    ext = FeatureExtractor(
        config=FeatureExtractorConfig(),
        embedder=FakeEmbedder(),
        nli_model=FakeNLIModel(),
    )
    embs, meta = ext.extract_query_features([])
    assert embs.shape == (0, 384)
    assert len(meta) == 0

    rag_empty = ext.extract_rag_features([], [], [])
    assert len(rag_empty) == 0
    assert {
        "cosine_sim",
        "nli_entailment",
        "nli_contradiction",
        "nli_neutral",
        "entity_overlap",
        "length_ratio",
        "response_token_count",
        "context_token_count",
    }.issubset(rag_empty.columns)

    with pytest.raises(ValueError, match="same length"):
        ext.extract_rag_features(["q1"], ["c1", "c2"], ["r1"])


def test_build_feature_tables_roundtrip(tmp_path: Path):
    rag_in = tmp_path / "ragtruth.parquet"
    rtr_in = tmp_path / "routerbench.parquet"
    out_dir = tmp_path / "processed"

    pd.DataFrame(
        {
            "id": ["1", "2"],
            "task_type": ["QA", "Summary"],
            "split": ["train", "test"],
            "prompt": ["What is France?", "Summarize Paris."],
            "context": ["Paris is in France.", "Paris has the Eiffel Tower."],
            "response": ["France is in Europe.", "Paris features the Eiffel Tower."],
            "has_hallucination": [True, False],
        }
    ).to_parquet(rag_in, index=False)

    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "eval_name": ["mmlu", "mbpp"],
            "prompt": ["What is 2+2?", "Write sort() in Python."],
            "split": ["train", "ood_test"],
            "tier1_quality": [1.0, 0.4],
            "tier2_quality": [1.0, 0.9],
            "tier1_cost": [0.001, 0.001],
            "tier2_cost": [0.03, 0.03],
            "tier1_sufficient": [True, False],
        }
    ).to_parquet(rtr_in, index=False)

    ext = FeatureExtractor(embedder=FakeEmbedder(), nli_model=FakeNLIModel())
    rag_out, rtr_out = build_feature_tables(rag_in, rtr_in, out_dir, extractor=ext)

    assert rag_out.exists()
    assert rtr_out.exists()

    rag_df = pd.read_parquet(rag_out)
    assert len(rag_df) == 2
    assert not rag_df.isna().any().any()
    assert {"id", "task_type", "split", "prompt", "context", "response", "has_hallucination"}.issubset(
        rag_df.columns
    )
    assert {"cosine_sim", "nli_entailment", "entity_overlap", "char_len", "avg_word_len"}.issubset(
        rag_df.columns
    )

    rtr_df = pd.read_parquet(rtr_out)
    assert len(rtr_df) == 2
    assert not rtr_df.isna().any().any()
    assert {"sample_id", "eval_name", "prompt", "split", "tier1_sufficient", "query_embedding"}.issubset(
        rtr_df.columns
    )
    assert len(rtr_df["query_embedding"].iloc[0]) == 384
