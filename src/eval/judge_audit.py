"""Bias-corrected LLM-as-a-judge audit for RAGTruth hallucination evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml
from dotenv import dotenv_values
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

MAX_JUDGE_SAMPLE_SIZE = 300
POSITION_SWAP_MARKER = "[ORDER: RESPONSE_FIRST]"
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")

_PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.1-8b-instant",
    },
}


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Spearman correlation without NaN/constant-input warnings."""
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0, 1.0
    res = spearmanr(x, y)
    r = 0.0 if np.isnan(res.statistic) else float(res.statistic)
    p = 1.0 if np.isnan(res.pvalue) else float(res.pvalue)
    return r, p


def _safe_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cohen's kappa safely when classes are degenerate."""
    if len(y_true) == 0:
        return 0.0
    if len(np.unique(y_true)) < 2 and len(np.unique(y_pred)) < 2:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    kappa = cohen_kappa_score(y_true, y_pred, labels=[0, 1])
    return 0.0 if np.isnan(kappa) else float(kappa)


def _compute_binary_agreement(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    spearman_r, spearman_p = _safe_spearman(y_true, scores)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "n_samples": len(y_true),
        "cohens_kappa": _safe_kappa(y_true, y_pred),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "accuracy": float(np.mean(y_true == y_pred)),
        "confusion_matrix": cm,
    }


def compute_judge_metrics(
    human_labels: Sequence[int],
    judge_scores: Sequence[float],
    task_types: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute overall and per-task agreement metrics between human labels and judge scores."""
    n = len(human_labels)
    if n == 0:
        raise ValueError("human_labels, judge_scores, and task_types must be non-empty")
    if len(judge_scores) != n or len(task_types) != n:
        raise ValueError("human_labels, judge_scores, and task_types must have equal length")

    y_true = np.asarray(human_labels, dtype=int)
    scores = np.asarray(judge_scores, dtype=float)
    tasks = np.asarray([str(t) for t in task_types])

    overall = _compute_binary_agreement(y_true, scores, threshold=threshold)
    by_task = {
        task: _compute_binary_agreement(
            y_true[tasks == task],
            scores[tasks == task],
            threshold=threshold,
        )
        for task in sorted(set(tasks.tolist()))
    }
    return {
        **overall,
        "threshold": float(threshold),
        "by_task": by_task,
    }


def fit_bias_correction(
    judge_scores: np.ndarray,
    response_lengths: np.ndarray,
    positions: np.ndarray,
    human_labels: np.ndarray,
    alpha: float = 1.0,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Estimate verbosity and position bias with Ridge regression and compute de-biased agreement."""
    scores = np.asarray(judge_scores, dtype=np.float64)
    lengths = np.asarray(response_lengths, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    y = np.asarray(human_labels, dtype=np.float64)

    n = len(scores)
    if n == 0 or not (len(lengths) == len(pos) == len(y) == n):
        raise ValueError("All input arrays must be non-empty and have equal length")

    len_mean = float(np.mean(lengths))
    len_std = float(np.std(lengths))
    pos_mean = float(np.mean(pos))
    pos_std = float(np.std(pos))

    len_scaled = (lengths - len_mean) / len_std if len_std > 0.0 else np.zeros_like(lengths)
    pos_scaled = (pos - pos_mean) / pos_std if pos_std > 0.0 else np.zeros_like(pos)

    design = np.column_stack([y, len_scaled, pos_scaled])
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design, scores)

    verbosity_coef = float(model.coef_[1] / len_std) if len_std > 0.0 else 0.0
    position_coef = float(model.coef_[2] / pos_std) if pos_std > 0.0 else 0.0

    corrected = apply_bias_correction(
        scores,
        lengths,
        pos,
        {
            "verbosity_coef": verbosity_coef,
            "position_coef": position_coef,
            "length_mean": len_mean,
            "position_mean": pos_mean,
        },
    )

    y_int = y.astype(int)
    raw_r, _ = _safe_spearman(y_int, scores)
    corr_r, _ = _safe_spearman(y_int, corrected)
    raw_kappa = _safe_kappa(y_int, (scores >= threshold).astype(int))
    corr_kappa = _safe_kappa(y_int, (corrected >= threshold).astype(int))

    return {
        "n_samples": int(n),
        "verbosity_coef": verbosity_coef,
        "position_coef": position_coef,
        "human_label_coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "length_mean": len_mean,
        "position_mean": pos_mean,
        "raw_spearman_r": raw_r,
        "corrected_spearman_r": corr_r,
        "raw_cohens_kappa": raw_kappa,
        "corrected_cohens_kappa": corr_kappa,
    }


def apply_bias_correction(
    judge_scores: Sequence[float] | np.ndarray,
    response_lengths: Sequence[float] | np.ndarray,
    positions: Sequence[float] | np.ndarray,
    bias_params: dict[str, float],
) -> np.ndarray:
    """Subtract estimated verbosity and position effects and clip scores to [0, 1]."""
    scores = np.asarray(judge_scores, dtype=np.float64)
    lengths = np.asarray(response_lengths, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    adjustment = (
        bias_params["verbosity_coef"] * (lengths - bias_params["length_mean"])
        + bias_params["position_coef"] * (pos - bias_params["position_mean"])
    )
    return np.clip(scores - adjustment, 0.0, 1.0)


def expand_response_for_verbosity(response: str) -> str:
    """Pad response with neutral non-factual framing to test length bias."""
    return (
        "To answer the prompt clearly based on the provided context: "
        f"{response} "
        "This completes the response strictly following the information available above."
    )


def swap_prompt_position(prompt: str) -> str:
    """Annotate prompt so the judge places Response before Context."""
    if prompt.startswith(POSITION_SWAP_MARKER):
        return prompt
    return f"{POSITION_SWAP_MARKER}\n{prompt}"


def build_judge_messages(prompt: str, context: str, response: str) -> list[dict[str, str]]:
    """Build OpenAI-compatible judge messages honoring the position-swap marker."""
    swapped = prompt.startswith(POSITION_SWAP_MARKER)
    clean_prompt = prompt.removeprefix(POSITION_SWAP_MARKER).lstrip("\n") if swapped else prompt

    sections = (
        [f"Response:\n{response}", f"Prompt:\n{clean_prompt}", f"Context:\n{context}"]
        if swapped
        else [f"Context:\n{context}", f"Prompt:\n{clean_prompt}", f"Response:\n{response}"]
    )
    user_content = (
        "Score whether the Response contains hallucinations (unsupported claims or "
        "contradictions relative to the Context) on a continuous scale from 0.0 "
        "(fully faithful) to 1.0 (clear hallucination).\n"
        'Respond strictly with JSON: {"score": <float between 0.0 and 1.0>}.\n\n'
        + "\n\n".join(sections)
    )
    return [
        {
            "role": "system",
            "content": "You are a strict factual consistency evaluator. Output valid JSON only.",
        },
        {"role": "user", "content": user_content},
    ]


def _parse_judge_score(content: str) -> float:
    """Extract a [0, 1] float score from JSON or plain-text judge output."""
    stripped = content.strip()
    candidates = [stripped]
    brace_match = re.search(r"\{.*?\}", stripped, flags=re.DOTALL)
    if brace_match:
        candidates.insert(0, brace_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "score" in parsed:
                return float(np.clip(float(parsed["score"]), 0.0, 1.0))
            if isinstance(parsed, (int, float)):
                return float(np.clip(float(parsed), 0.0, 1.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    match = _FLOAT_RE.search(stripped)
    if not match:
        raise ValueError(f"Unable to parse judge score from response: {stripped!r}")
    return float(np.clip(float(match.group(0)), 0.0, 1.0))


def _resolve_api_credentials(env_file: Path | None = Path(".env")) -> tuple[str, str] | None:
    """Resolve (provider, api_key) from os.environ or .env without fabricating credentials."""
    file_env: dict[str, str | None] = {}
    if env_file is not None and env_file.exists():
        file_env = dotenv_values(env_file)

    for provider, cfg in _PROVIDER_CONFIGS.items():
        env_var = cfg["env_var"]
        raw = os.environ.get(env_var)
        if not raw or not raw.strip():
            raw = file_env.get(env_var)
        if raw and raw.strip():
            return provider, raw.strip()
    return None


def _create_httpx_judge_fn(
    provider: str,
    api_key: str,
    client: httpx.AsyncClient,
) -> Callable[[str, str, str], Awaitable[float]]:
    cfg = _PROVIDER_CONFIGS[provider]

    async def _judge(prompt: str, context: str, response: str) -> float:
        payload = {
            "model": cfg["model"],
            "messages": build_judge_messages(prompt, context, response),
            "temperature": 0.0,
            "max_tokens": 64,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = await client.post(cfg["url"], json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_judge_score(content)

    return _judge


def _is_retryable_error(exc: Exception) -> tuple[bool, float | None]:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status in {429, 502, 503, 504}:
        retry_after = None
        headers = getattr(response, "headers", None)
        if headers:
            raw_ra = headers.get("Retry-After")
            if raw_ra is not None:
                try:
                    retry_after = max(0.0, float(raw_ra))
                except ValueError:
                    retry_after = None
        return True, retry_after
    if isinstance(exc, httpx.TransportError):
        return True, None
    return False, None


async def _call_with_retry(
    judge_fn: Callable[[str, str, str], Awaitable[float]],
    prompt: str,
    context: str,
    response: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> float:
    for attempt in range(max_retries + 1):
        try:
            score = await judge_fn(prompt, context, response)
            return float(np.clip(float(score), 0.0, 1.0))
        except Exception as exc:
            retryable, retry_after = _is_retryable_error(exc)
            if not retryable or attempt >= max_retries:
                raise
            delay = retry_after if retry_after is not None else base_delay * (2**attempt)
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable retry exhaustion")


def _stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")

    target_n = min(sample_size, MAX_JUDGE_SAMPLE_SIZE, len(df))
    if target_n >= len(df):
        return df.copy().reset_index(drop=True)

    strat_joint = df["task_type"].astype(str) + "__" + df["has_hallucination"].astype(int).astype(str)
    for strat_col in (strat_joint, df["task_type"].astype(str)):
        counts = strat_col.value_counts()
        if len(counts) > 1 and counts.min() >= 2 and target_n >= len(counts):
            sampled_idx, _ = train_test_split(
                df.index.to_numpy(),
                train_size=target_n,
                stratify=strat_col,
                random_state=random_state,
            )
            return df.loc[sampled_idx].reset_index(drop=True)

    return df.sample(n=target_n, random_state=random_state).reset_index(drop=True)


def _load_checkpoint(checkpoint_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return records
    for line in checkpoint_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            # Ignore a partial trailing line written during an interrupted run.
            continue
        records[(str(rec["id"]), str(rec["variant"]))] = rec
    return records


def _build_eval_tasks(sampled: pd.DataFrame) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in sampled.itertuples(index=False):
        item_id = str(row.id)
        prompt = "" if pd.isna(row.prompt) else str(row.prompt)
        context = "" if pd.isna(row.context) else str(row.context)
        response = "" if pd.isna(row.response) else str(row.response)
        human_label = int(bool(row.has_hallucination))
        task_type = str(row.task_type)

        expanded = expand_response_for_verbosity(response)
        swapped = swap_prompt_position(prompt)

        variants = [
            ("base", prompt, response, len(response), 0),
            ("length_expanded", prompt, expanded, len(expanded), 0),
            ("position_swapped", swapped, response, len(response), 1),
        ]
        for variant, v_prompt, v_response, resp_len, pos in variants:
            tasks.append(
                {
                    "id": item_id,
                    "variant": variant,
                    "prompt": v_prompt,
                    "context": context,
                    "response": v_response,
                    "response_length": resp_len,
                    "position": pos,
                    "human_label": human_label,
                    "task_type": task_type,
                }
            )
    return tasks


async def _evaluate_with_checkpoints(
    tasks: list[dict[str, Any]],
    judge_fn: Callable[[str, str, str], Awaitable[float]],
    checkpoint_path: Path,
    max_concurrency: int,
    max_retries: int,
    retry_base_delay: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    records = _load_checkpoint(checkpoint_path)
    pending = [t for t in tasks if (t["id"], t["variant"]) not in records]
    if not pending:
        return records

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    write_lock = asyncio.Lock()

    async def _run_one(task: dict[str, Any]) -> None:
        async with semaphore:
            score = await _call_with_retry(
                judge_fn,
                task["prompt"],
                task["context"],
                task["response"],
                max_retries=max_retries,
                base_delay=retry_base_delay,
            )
        record = {
            "id": task["id"],
            "variant": task["variant"],
            "score": score,
            "response_length": task["response_length"],
            "position": task["position"],
            "human_label": task["human_label"],
            "task_type": task["task_type"],
        }
        async with write_lock:
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            records[(record["id"], record["variant"])] = record

    await asyncio.gather(*(_run_one(task) for task in pending))
    return records


def _run_coro_sync(coro: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def run_judge_audit(
    features_parquet: Path,
    out_json: Path,
    sample_size: int = 300,
    judge_fn: Callable[[str, str, str], Awaitable[float]] | None = None,
    *,
    random_state: int = 42,
    checkpoint_path: Path | None = None,
    max_concurrency: int = 8,
    max_retries: int = 5,
    retry_base_delay: float = 1.0,
) -> dict[str, Any]:
    """Run stratified paired LLM-as-a-judge audit and write JSON summary artifact."""
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    creds = _resolve_api_credentials() if judge_fn is None else None
    if judge_fn is None and creds is None:
        tbd_payload: dict[str, Any] = {
            "status": "tbd",
            "reason": (
                "No DEEPSEEK_API_KEY or GROQ_API_KEY configured in environment or .env, "
                "and no judge_fn provided."
            ),
            "sample_size": 0,
            "requested_sample_size": int(min(sample_size, MAX_JUDGE_SAMPLE_SIZE)),
            "features_parquet": str(features_parquet),
            "metrics": None,
            "corrected_metrics": None,
            "bias_correction": None,
        }
        out_path.write_text(json.dumps(tbd_payload, indent=2) + "\n", encoding="utf-8")
        return tbd_payload

    df = pd.read_parquet(features_parquet)
    required = ["id", "task_type", "prompt", "context", "response", "has_hallucination"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Features parquet missing required columns: {missing}")

    sampled = _stratified_sample(df, sample_size=sample_size, random_state=random_state)
    eval_tasks = _build_eval_tasks(sampled)
    ckpt = checkpoint_path or out_path.with_suffix(".checkpoint.jsonl")

    async def _execute() -> dict[tuple[str, str], dict[str, Any]]:
        if judge_fn is not None:
            return await _evaluate_with_checkpoints(
                eval_tasks,
                judge_fn,
                ckpt,
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
        assert creds is not None
        provider, api_key = creds
        async with httpx.AsyncClient() as client:
            http_judge = _create_httpx_judge_fn(provider, api_key, client)
            return await _evaluate_with_checkpoints(
                eval_tasks,
                http_judge,
                ckpt,
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )

    records = _run_coro_sync(_execute())

    base_records = [records[(str(row_id), "base")] for row_id in sampled["id"]]
    all_records = [records[(t["id"], t["variant"])] for t in eval_tasks]

    base_human = [int(r["human_label"]) for r in base_records]
    base_scores = [float(r["score"]) for r in base_records]
    base_tasks = [str(r["task_type"]) for r in base_records]
    metrics = compute_judge_metrics(base_human, base_scores, base_tasks)

    bias_res = fit_bias_correction(
        judge_scores=np.array([r["score"] for r in all_records], dtype=float),
        response_lengths=np.array([r["response_length"] for r in all_records], dtype=float),
        positions=np.array([r["position"] for r in all_records], dtype=float),
        human_labels=np.array([r["human_label"] for r in all_records], dtype=float),
    )

    corrected_base_scores = apply_bias_correction(
        base_scores,
        [float(r["response_length"]) for r in base_records],
        [float(r["position"]) for r in base_records],
        bias_res,
    )
    corrected_metrics = compute_judge_metrics(
        base_human,
        corrected_base_scores.tolist(),
        base_tasks,
    )

    summary: dict[str, Any] = {
        "status": "ready",
        "provider": creds[0] if creds is not None else "injected",
        "sample_size": len(sampled),
        "paired_evaluations": len(all_records),
        "features_parquet": str(features_parquet),
        "checkpoint_path": str(ckpt),
        "metrics": metrics,
        "corrected_metrics": corrected_metrics,
        "bias_correction": bias_res,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bias-corrected LLM-as-a-judge audit.")
    parser.add_argument("--config", type=Path, default=Path("config/ragtruth_conformal.yaml"))
    parser.add_argument(
        "--features-parquet",
        type=Path,
        default=Path("data/processed/ragtruth_features.parquet"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/judge_audit_summary.json"),
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    sample_size = MAX_JUDGE_SAMPLE_SIZE
    random_state = 42
    if args.config.exists():
        raw_cfg = yaml.safe_load(args.config.read_text()) or {}
        if raw_cfg.get("judge_sample_size") is not None:
            sample_size = int(raw_cfg["judge_sample_size"])
        if raw_cfg.get("random_state") is not None:
            random_state = int(raw_cfg["random_state"])
    if args.sample_size is not None:
        sample_size = args.sample_size

    summary = run_judge_audit(
        features_parquet=args.features_parquet,
        out_json=args.out_json,
        sample_size=sample_size,
        random_state=random_state,
        checkpoint_path=args.checkpoint,
    )
    print(f"Judge audit status={summary['status']} sample_size={summary['sample_size']} -> {args.out_json}")


if __name__ == "__main__":
    main()
