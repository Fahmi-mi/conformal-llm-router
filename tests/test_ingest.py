import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import ingest
from src.eval.ingest import (
    _parse_hallucination_flag,
    ingest_ragtruth,
    ingest_routerbench,
    normalize_ragtruth_df,
    normalize_routerbench_df,
)


def test_normalize_ragtruth_deterministic_splits_and_schema():
    raw = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(10)],
            "task_type": ["QA"] * 5 + ["Summary"] * 5,
            "split": ["train"] * 8 + ["test"] * 2,
            "prompt": ["q?"] * 10,
            "context": ["ctx"] * 10,
            "response": ["ans"] * 10,
            "labels": [[]] * 7 + [[{"label_type": "Evident Conflict"}]] * 3,
        }
    )
    df1 = normalize_ragtruth_df(raw, cal_ratio=0.25, random_state=42)
    df2 = normalize_ragtruth_df(raw, cal_ratio=0.25, random_state=42)
    assert df1["split"].tolist() == df2["split"].tolist()
    assert set(df1["split"]) == {"train", "cal", "test"}
    assert df1["has_hallucination"].dtype == bool


def test_normalize_ragtruth_missing_column_raises():
    with pytest.raises(ValueError, match="missing required"):
        normalize_ragtruth_df(pd.DataFrame({"id": ["1"]}))


def test_normalize_routerbench_ood_holdout_and_tiers():
    raw = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "eval_name": ["mmlu", "gsm8k", "mbpp", "hellaswag"],
            "prompt": ["p1", "p2", "p3", "p4"],
            "mistralai/mistral-7b-chat": [0.9, 0.2, 0.8, 0.7],
            "gpt-4-1106-preview": [0.92, 0.9, 0.85, 0.75],
            "mistralai/mistral-7b-chat|total_cost": [0.001, 0.001, 0.001, 0.001],
            "gpt-4-1106-preview|total_cost": [0.03, 0.03, 0.03, 0.03],
        }
    )
    df = normalize_routerbench_df(raw, ood_tasks=["mbpp"], random_state=42)
    assert df.loc[df["eval_name"] == "mbpp", "split"].iloc[0] == "ood_test"
    assert {
        "tier1_quality",
        "tier2_quality",
        "tier1_cost",
        "tier2_cost",
        "tier1_sufficient",
    }.issubset(df.columns)


def test_normalize_ragtruth_wandb_schema_variant_and_non_range_index():
    raw = pd.DataFrame(
        {
            "id": [f"w{i}" for i in range(10)],
            "task_type": ["QA"] * 5 + ["Data2txt"] * 5,
            "split": ["train"] * 8 + ["test"] * 2,
            "query": [f"q{i}" for i in range(10)],
            "context": ["ctx"] * 10,
            "output": ["ans"] * 10,
            "hallucination_labels": ["[]"] * 7 + ['[{"meta": "EVIDENT CONFLICT"}]'] * 3,
            "hallucination_labels_processed": [
                {"evident_conflict": 0, "baseless_info": 0}
            ] * 7
            + [{"evident_conflict": 1, "baseless_info": 0}] * 3,
        },
        index=list(range(100, 110)),
    )
    df = normalize_ragtruth_df(raw, cal_ratio=0.25, random_state=42)
    assert {"id", "task_type", "split", "prompt", "context", "response", "has_hallucination"}.issubset(
        df.columns
    )
    assert df["has_hallucination"].tolist() == [False] * 7 + [True] * 3
    assert set(df["split"]) == {"train", "cal", "test"}


def test_normalize_ragtruth_invalid_cal_ratio_raises():
    raw = pd.DataFrame(
        {
            "id": ["1"],
            "task_type": ["QA"],
            "split": ["train"],
            "prompt": ["p"],
            "context": ["c"],
            "response": ["r"],
            "labels": [[]],
        }
    )
    with pytest.raises(ValueError, match="cal_ratio"):
        normalize_ragtruth_df(raw, cal_ratio=0.0)


def test_normalize_routerbench_missing_column_raises():
    with pytest.raises(ValueError, match="missing required"):
        normalize_routerbench_df(pd.DataFrame({"sample_id": ["s1"]}), ood_tasks=["mbpp"])


def test_normalize_routerbench_deterministic_splits_and_sufficiency():
    raw = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(10)],
            "eval_name": ["mmlu"] * 4 + ["gsm8k"] * 4 + ["mbpp"] * 2,
            "prompt": [f"p{i}" for i in range(10)],
            "mistralai/mistral-7b-chat": [0.9, 0.2, 0.8, 0.7] * 2 + [0.5, 0.9],
            "gpt-4-1106-preview": [0.92, 0.9, 0.85, 0.75] * 2 + [0.9, 0.9],
            "mistralai/mistral-7b-chat|total_cost": [0.001] * 10,
            "gpt-4-1106-preview|total_cost": [0.03] * 10,
        },
        index=list(range(20, 30)),
    )
    df1 = normalize_routerbench_df(raw, ood_tasks=["mbpp"], random_state=42)
    df2 = normalize_routerbench_df(raw, ood_tasks=["mbpp"], random_state=42)
    assert df1["split"].tolist() == df2["split"].tolist()
    assert set(df1["split"]) == {"train", "cal", "test", "ood_test"}
    assert df1["tier1_sufficient"].dtype == bool
    assert bool(df1["tier1_sufficient"].iloc[0]) is True
    assert bool(df1["tier1_sufficient"].iloc[1]) is False
    assert bool(df1["tier1_sufficient"].iloc[3]) is True


def test_normalize_routerbench_rejects_nan_and_nonfinite():
    base = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "eval_name": ["mmlu", "gsm8k", "mbpp", "hellaswag"],
            "prompt": ["p1", "p2", "p3", "p4"],
            "mistralai/mistral-7b-chat": [0.9, np.nan, 0.8, 0.7],
            "gpt-4-1106-preview": [0.92, 0.9, 0.85, 0.75],
            "mistralai/mistral-7b-chat|total_cost": [0.001, 0.001, 0.001, 0.001],
            "gpt-4-1106-preview|total_cost": [0.03, 0.03, np.inf, 0.03],
        }
    )
    with pytest.raises(ValueError, match=r"mistralai/mistral-7b-chat.*1.*gpt-4-1106-preview\|total_cost.*1"):
        normalize_routerbench_df(base, ood_tasks=["mbpp"], random_state=42)


def test_normalize_routerbench_ood_tasks_zero_match_raises():
    raw = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "eval_name": ["mmlu", "gsm8k", "hellaswag"],
            "prompt": ["p1", "p2", "p3"],
            "mistralai/mistral-7b-chat": [0.9, 0.2, 0.7],
            "gpt-4-1106-preview": [0.92, 0.9, 0.75],
            "mistralai/mistral-7b-chat|total_cost": [0.001, 0.001, 0.001],
            "gpt-4-1106-preview|total_cost": [0.03, 0.03, 0.03],
        }
    )
    with pytest.raises(ValueError, match="nonexistent_task"):
        normalize_routerbench_df(raw, ood_tasks=["nonexistent_task"], random_state=42)


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ({"evident_conflict": None, "baseless_info": 0}, False),
        ({"evident_conflict": 0, "baseless_info": None}, False),
        ({"evident_conflict": 1, "baseless_info": None}, True),
        ({"evident_conflict": 0, "baseless_info": 2}, True),
        ({"other": 0}, False),
        ({"other": 1}, True),
        ({}, False),
        ('{"evident_conflict": 0, "baseless_info": 0}', False),
        ('{"evident_conflict": 1, "baseless_info": 0}', True),
        ("[]", False),
        ('[{"meta": "EVIDENT CONFLICT"}]', True),
        ("false", False),
        ("0", False),
        ("null", False),
        ("", False),
        ('""', False),
        ("true", True),
        ("1", True),
        ("Evident Conflict", True),
        (None, False),
        (float("nan"), False),
    ],
)
def test_parse_hallucination_flag_edge_cases(val, expected):
    assert _parse_hallucination_flag(val) is expected


class _FakeSplit:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


def _sample_ragtruth_splits() -> dict[str, _FakeSplit]:
    train_df = pd.DataFrame(
        {
            "id": [f"tr{i}" for i in range(8)],
            "task_type": ["QA"] * 4 + ["Summary"] * 4,
            "prompt": [f"q{i}" for i in range(8)],
            "context": ["ctx"] * 8,
            "response": ["ans"] * 8,
            "labels": [[]] * 6 + [[{"label_type": "Evident Conflict"}]] * 2,
        }
    )
    test_df = pd.DataFrame(
        {
            "id": ["te0", "te1"],
            "task_type": ["QA", "Summary"],
            "prompt": ["qt0", "qt1"],
            "context": ["ctx", "ctx"],
            "response": ["ans", "ans"],
            "labels": [[], [{"label_type": "Baseless Info"}]],
        }
    )
    return {"train": _FakeSplit(train_df), "test": _FakeSplit(test_df)}


def _sample_routerbench_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(10)],
            "eval_name": ["mmlu"] * 4 + ["gsm8k"] * 4 + ["mbpp"] * 2,
            "prompt": [f"p{i}" for i in range(10)],
            "mistralai/mistral-7b-chat": [0.9, 0.2, 0.8, 0.7] * 2 + [0.5, 0.9],
            "gpt-4-1106-preview": [0.92, 0.9, 0.85, 0.75] * 2 + [0.9, 0.9],
            "mistralai/mistral-7b-chat|total_cost": [0.001] * 10,
            "gpt-4-1106-preview|total_cost": [0.03] * 10,
        }
    )


def test_ingest_ragtruth_and_routerbench_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ingest, "load_dataset", lambda repo: _sample_ragtruth_splits())
    rag_out = ingest_ragtruth(tmp_path / "raw/ragtruth.parquet", cal_ratio=0.25, random_state=42)
    rag_df = pd.read_parquet(rag_out)
    assert len(rag_df) == 10
    assert set(rag_df["split"]) == {"train", "cal", "test"}

    pkl_path = tmp_path / "routerbench_0shot.pkl"
    _sample_routerbench_df().to_pickle(pkl_path)
    monkeypatch.setattr(ingest, "hf_hub_download", lambda **kwargs: str(pkl_path))
    rtr_out = ingest_routerbench(tmp_path / "raw/routerbench.parquet", ood_tasks=["mbpp"], random_state=42)
    rtr_df = pd.read_parquet(rtr_out)
    assert len(rtr_df) == 10
    assert set(rtr_df["split"]) == {"train", "cal", "test", "ood_test"}


def test_ingest_cli_main_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    rag_cfg = tmp_path / "ragtruth.yaml"
    rag_cfg.write_text("cal_ratio: 0.25\nrandom_state: 42\n")
    rtr_cfg = tmp_path / "routerbench.yaml"
    rtr_cfg.write_text("ood_holdout_tasks:\n  - mbpp\nrandom_state: 42\n")

    rag_out = tmp_path / "out/ragtruth.parquet"
    rtr_out = tmp_path / "out/routerbench.parquet"

    pkl_path = tmp_path / "routerbench_0shot.pkl"
    _sample_routerbench_df().to_pickle(pkl_path)
    monkeypatch.setattr(ingest, "load_dataset", lambda repo: _sample_ragtruth_splits())
    monkeypatch.setattr(ingest, "hf_hub_download", lambda **kwargs: str(pkl_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest",
            "--ragtruth-config",
            str(rag_cfg),
            "--routerbench-config",
            str(rtr_cfg),
            "--ragtruth-out",
            str(rag_out),
            "--routerbench-out",
            str(rtr_out),
            "--only",
            "all",
        ],
    )

    ingest.main()
    captured = capsys.readouterr().out
    assert rag_out.exists()
    assert rtr_out.exists()
    assert "Saved RAGTruth" in captured
    assert "Saved RouterBench" in captured
