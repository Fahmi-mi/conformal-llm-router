"""Benchmark dataset ingestion and deterministic splitting for RAGTruth and RouterBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

RAGTRUTH_HF_REPO = "wandb/RAGTruth-processed"
ROUTERBENCH_HF_REPO = "withmartian/routerbench"
ROUTERBENCH_FILENAME = "routerbench_0shot.pkl"

TIER1_MODEL = "mistralai/mistral-7b-chat"
TIER2_MODEL = "gpt-4-1106-preview"
TIER1_COST_COL = f"{TIER1_MODEL}|total_cost"
TIER2_COST_COL = f"{TIER2_MODEL}|total_cost"


def _require_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


def _parse_hallucination_flag(val: Any) -> bool:
    if isinstance(val, dict):
        if "evident_conflict" in val or "baseless_info" in val:
            return any(
                (
                    float(raw) > 0
                    if isinstance(raw, (int, float, bool, np.integer, np.floating, np.bool_))
                    else _parse_hallucination_flag(raw)
                )
                for raw in (val.get("evident_conflict") or 0, val.get("baseless_info") or 0)
            )
        return any(_parse_hallucination_flag(v) for v in val.values())
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return False
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped.lower() not in {"false", "none", "null", "nan", "0"}
        if isinstance(parsed, str) and parsed == stripped:
            return True
        return _parse_hallucination_flag(parsed)
    if isinstance(val, (list, tuple, set, np.ndarray)):
        return len(val) > 0
    if pd.isna(val):
        return False
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, (bool, np.bool_)):
        return float(val) > 0
    return bool(val)


def _stratified_or_random_split(
    indices: np.ndarray,
    test_size: int,
    stratify_labels: pd.Series | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    strat = None
    if stratify_labels is not None:
        counts = stratify_labels.value_counts()
        n_train = len(indices) - test_size
        if len(counts) > 1 and counts.min() >= 2 and test_size >= len(counts) and n_train >= len(counts):
            strat = stratify_labels
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )


def normalize_ragtruth_df(
    raw_df: pd.DataFrame,
    cal_ratio: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Normalize a RAGTruth DataFrame and deterministically split train into train/cal."""
    if not (0.0 < cal_ratio < 1.0):
        raise ValueError(f"cal_ratio must be in (0, 1), got {cal_ratio}")

    df = raw_df.copy()
    if "prompt" not in df.columns and "query" in df.columns:
        df["prompt"] = df["query"]
    if "response" not in df.columns and "output" in df.columns:
        df["response"] = df["output"]

    _require_columns(df, ["id", "task_type", "split", "prompt", "context", "response"], "RAGTruth")

    if "hallucination_labels_processed" in df.columns:
        has_hallucination = df["hallucination_labels_processed"].apply(_parse_hallucination_flag)
    elif "labels" in df.columns:
        has_hallucination = df["labels"].apply(_parse_hallucination_flag)
    elif "hallucination_labels" in df.columns:
        has_hallucination = df["hallucination_labels"].apply(_parse_hallucination_flag)
    else:
        raise ValueError(
            "RAGTruth missing required label column: expected one of "
            "['labels', 'hallucination_labels', 'hallucination_labels_processed']"
        )

    out = pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "task_type": df["task_type"].astype(str),
            "split": df["split"].astype(str),
            "prompt": df["prompt"].fillna("").astype(str),
            "context": df["context"].fillna("").astype(str),
            "response": df["response"].fillna("").astype(str),
            "has_hallucination": has_hallucination.astype(bool),
        }
    ).reset_index(drop=True)

    train_mask = out["split"] == "train"
    train_idx = np.flatnonzero(train_mask.to_numpy())
    if len(train_idx) >= 2:
        strat_key = out.loc[train_idx, "task_type"] + "_" + out.loc[train_idx, "has_hallucination"].astype(str)
        n_cal = max(1, round(len(train_idx) * cal_ratio))
        if n_cal >= len(train_idx):
            n_cal = len(train_idx) - 1
        _, cal_idx = _stratified_or_random_split(
            train_idx,
            test_size=n_cal,
            stratify_labels=strat_key,
            random_state=random_state,
        )
        out.loc[cal_idx, "split"] = "cal"

    return out


def normalize_routerbench_df(
    raw_df: pd.DataFrame,
    ood_tasks: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Normalize a RouterBench DataFrame, hold out OOD tasks, and create deterministic splits."""
    required = [
        "sample_id",
        "eval_name",
        "prompt",
        TIER1_MODEL,
        TIER2_MODEL,
        TIER1_COST_COL,
        TIER2_COST_COL,
    ]
    _require_columns(raw_df, required, "RouterBench")
    df = raw_df.reset_index(drop=True)

    t1_q = pd.to_numeric(df[TIER1_MODEL], errors="raise").astype(float)
    t2_q = pd.to_numeric(df[TIER2_MODEL], errors="raise").astype(float)
    t1_c = pd.to_numeric(df[TIER1_COST_COL], errors="raise").astype(float)
    t2_c = pd.to_numeric(df[TIER2_COST_COL], errors="raise").astype(float)

    bad_cols = {
        col: int((~np.isfinite(series)).sum())
        for col, series in (
            (TIER1_MODEL, t1_q),
            (TIER2_MODEL, t2_q),
            (TIER1_COST_COL, t1_c),
            (TIER2_COST_COL, t2_c),
        )
        if (~np.isfinite(series)).any()
    }
    if bad_cols:
        raise ValueError(
            f"RouterBench contains NaN or non-finite values in numeric columns: {bad_cols}"
        )

    ood_set = set(ood_tasks)
    eval_names = df["eval_name"].astype(str)
    splits = pd.Series("train", index=df.index, dtype="object")
    ood_mask = eval_names.isin(ood_set)
    if ood_tasks and not ood_mask.any():
        raise ValueError(f"ood_tasks {ood_tasks!r} matched 0 rows in RouterBench eval_name")
    splits.loc[ood_mask] = "ood_test"

    in_dist_idx = np.flatnonzero((~ood_mask).to_numpy())
    if len(in_dist_idx) >= 3:
        n_test = max(1, round(len(in_dist_idx) * 0.2))
        n_cal = max(1, round(len(in_dist_idx) * 0.2))
        if n_test + n_cal >= len(in_dist_idx):
            n_test = 1
            n_cal = 1
        train_cal_idx, test_idx = _stratified_or_random_split(
            in_dist_idx,
            test_size=n_test,
            stratify_labels=eval_names.iloc[in_dist_idx],
            random_state=random_state,
        )
        _, cal_idx = _stratified_or_random_split(
            train_cal_idx,
            test_size=n_cal,
            stratify_labels=eval_names.iloc[train_cal_idx],
            random_state=random_state,
        )
        splits.iloc[test_idx] = "test"
        splits.iloc[cal_idx] = "cal"

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "eval_name": eval_names,
            "prompt": df["prompt"].fillna("").astype(str),
            "split": splits.astype(str),
            "tier1_quality": t1_q,
            "tier2_quality": t2_q,
            "tier1_cost": t1_c,
            "tier2_cost": t2_c,
            "tier1_sufficient": ((t2_q - t1_q) <= 0.05 + 1e-9).astype(bool),
        }
    ).reset_index(drop=True)
    return out


def ingest_ragtruth(
    output_path: Path,
    cal_ratio: float = 0.2,
    random_state: int = 42,
) -> Path:
    """Download RAGTruth from Hugging Face, normalize splits, and save as Parquet."""
    ds = load_dataset(RAGTRUTH_HF_REPO)
    frames = []
    for split_name in ds:
        frame = ds[split_name].to_pandas()
        frame["split"] = split_name
        frames.append(frame)
    raw_df = pd.concat(frames, ignore_index=True)

    normalized = normalize_ragtruth_df(raw_df, cal_ratio=cal_ratio, random_state=random_state)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(out, index=False)
    return out


def ingest_routerbench(
    output_path: Path,
    ood_tasks: list[str],
    random_state: int = 42,
) -> Path:
    """Download RouterBench 0-shot pickle from Hugging Face, normalize, and save as Parquet."""
    pkl_path = hf_hub_download(
        repo_id=ROUTERBENCH_HF_REPO,
        filename=ROUTERBENCH_FILENAME,
        repo_type="dataset",
    )
    raw_df = pd.read_pickle(pkl_path)
    normalized = normalize_routerbench_df(raw_df, ood_tasks=ood_tasks, random_state=random_state)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RAGTruth and RouterBench datasets.")
    parser.add_argument("--ragtruth-config", type=Path, default=Path("config/ragtruth_conformal.yaml"))
    parser.add_argument("--routerbench-config", type=Path, default=Path("config/routerbench.yaml"))
    parser.add_argument("--ragtruth-out", type=Path, default=Path("data/raw/ragtruth.parquet"))
    parser.add_argument("--routerbench-out", type=Path, default=Path("data/raw/routerbench.parquet"))
    parser.add_argument("--only", choices=["all", "ragtruth", "routerbench"], default="all")
    args = parser.parse_args()

    if args.only in ("all", "ragtruth"):
        rag_cfg = yaml.safe_load(args.ragtruth_config.read_text())
        rag_path = ingest_ragtruth(
            args.ragtruth_out,
            cal_ratio=float(rag_cfg["cal_ratio"]),
            random_state=int(rag_cfg["random_state"]),
        )
        rag_df = pd.read_parquet(rag_path)
        print(f"Saved RAGTruth -> {rag_path} ({len(rag_df)} rows, splits={rag_df['split'].value_counts().to_dict()})")

    if args.only in ("all", "routerbench"):
        rtr_cfg = yaml.safe_load(args.routerbench_config.read_text())
        rtr_path = ingest_routerbench(
            args.routerbench_out,
            ood_tasks=list(rtr_cfg["ood_holdout_tasks"]),
            random_state=int(rtr_cfg["random_state"]),
        )
        rtr_df = pd.read_parquet(rtr_path)
        print(f"Saved RouterBench -> {rtr_path} ({len(rtr_df)} rows, splits={rtr_df['split'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
