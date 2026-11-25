"""Tests for bias-corrected LLM-as-a-judge audit."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from src.eval.judge_audit import (
    _create_httpx_judge_fn,
    _parse_judge_score,
    build_judge_messages,
    compute_judge_metrics,
    fit_bias_correction,
    run_judge_audit,
    swap_prompt_position,
)


def test_compute_judge_metrics_kappa_and_spearman():
    human = [0, 0, 1, 1, 0, 1]
    judge = [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]
    tasks = ["QA", "QA", "QA", "Summary", "Summary", "Summary"]
    res = compute_judge_metrics(human, judge, tasks)
    assert res["cohens_kappa"] > 0.8
    assert res["spearman_r"] > 0.8
    assert set(res["by_task"].keys()) == {"QA", "Summary"}
    assert res["confusion_matrix"] == [[3, 0], [0, 3]]
    assert res["n_samples"] == 6


def test_compute_judge_metrics_handles_constant_inputs():
    res = compute_judge_metrics([0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1], ["QA"] * 4)
    assert np.isfinite(res["cohens_kappa"])
    assert np.isfinite(res["spearman_r"])
    assert res["confusion_matrix"] == [[4, 0], [0, 0]]


def test_fit_bias_correction_handles_constant_length_and_corrects_verbosity():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=80)
    lengths = rng.integers(20, 400, size=80)
    positions = rng.integers(0, 2, size=80)
    biased_scores = np.clip(0.2 + 0.5 * y + 0.001 * lengths + 0.1 * positions, 0, 1)
    res = fit_bias_correction(biased_scores, lengths, positions, y)
    assert "verbosity_coef" in res and "position_coef" in res
    assert res["verbosity_coef"] > 0.0
    assert res["position_coef"] > 0.0
    assert res["corrected_spearman_r"] >= res["raw_spearman_r"] - 0.05

    # Constant length and constant position edge cases
    res_const = fit_bias_correction(biased_scores, np.ones(80), positions, y)
    assert np.isfinite(res_const["verbosity_coef"])
    assert res_const["verbosity_coef"] == 0.0

    res_const_pos = fit_bias_correction(biased_scores, lengths, np.zeros(80), y)
    assert np.isfinite(res_const_pos["position_coef"])
    assert res_const_pos["position_coef"] == 0.0


def _make_sample_features_parquet(path: Path, n_per_task: int = 6) -> Path:
    rows = []
    idx = 0
    for task in ("QA", "Summary", "Data2txt"):
        for i in range(n_per_task):
            label = i % 2
            rows.append(
                {
                    "id": f"{task.lower()}_{idx}",
                    "task_type": task,
                    "split": "test",
                    "prompt": f"Question {idx} about context",
                    "context": f"Source facts for item {idx}.",
                    "response": f"Answer {idx} " + ("with extra detail " if label else "brief"),
                    "has_hallucination": bool(label),
                }
            )
            idx += 1
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def test_run_judge_audit_tbd_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    parquet_path = _make_sample_features_parquet(tmp_path / "features.parquet")
    out_json = tmp_path / "artifacts" / "judge_audit_summary.json"

    res = run_judge_audit(parquet_path, out_json, sample_size=12, judge_fn=None)
    assert res["status"] == "tbd"
    assert res["sample_size"] == 0
    assert res.get("reason")
    assert out_json.exists()
    written = json.loads(out_json.read_text())
    assert written["status"] == "tbd"
    assert written["sample_size"] == 0


def test_run_judge_audit_with_injected_judge_fn_stratified_and_paired(tmp_path: Path):
    parquet_path = _make_sample_features_parquet(tmp_path / "features.parquet", n_per_task=6)
    out_json = tmp_path / "artifacts" / "judge_audit_summary.json"

    recorded_calls: list[tuple[str, str, str]] = []

    async def biased_judge(prompt: str, context: str, response: str) -> float:
        recorded_calls.append((prompt, context, response))
        # Ground-truth signal encoded in response text in this synthetic fixture
        is_hallucinated = "with extra detail" in response
        pos_bias = 0.12 if "[ORDER: RESPONSE_FIRST]" in prompt else 0.0
        len_bias = min(0.25, 0.0015 * len(response))
        base = 0.65 if is_hallucinated else 0.15
        return float(np.clip(base + pos_bias + len_bias, 0.0, 1.0))

    res = run_judge_audit(parquet_path, out_json, sample_size=12, judge_fn=biased_judge)

    assert res["status"] == "ready"
    assert res["sample_size"] == 12
    # 3 paired evaluations per sampled row: base, length_expanded, position_swapped
    assert len(recorded_calls) == 36
    assert any("[ORDER: RESPONSE_FIRST]" in p for p, _, _ in recorded_calls)
    assert res["metrics"]["cohens_kappa"] >= 0.0
    assert set(res["metrics"]["by_task"].keys()) == {"Data2txt", "QA", "Summary"}
    assert res["bias_correction"]["verbosity_coef"] > 0.0
    assert res["bias_correction"]["position_coef"] > 0.0

    written = json.loads(out_json.read_text())
    assert written["status"] == "ready"
    assert written["sample_size"] == 12


def test_run_judge_audit_retries_429_and_resumes_from_checkpoint(tmp_path: Path):
    parquet_path = _make_sample_features_parquet(tmp_path / "features.parquet", n_per_task=2)
    out_json = tmp_path / "artifacts" / "judge_audit_summary.json"
    checkpoint_path = out_json.with_suffix(".checkpoint.jsonl")

    attempts = {"count": 0, "successful": 0}
    req = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    resp_429 = httpx.Response(429, request=req, headers={"Retry-After": "0.001"})

    async def flaky_then_crashing_judge(prompt: str, context: str, response: str) -> float:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp_429)
        if attempts["successful"] >= 4:
            raise RuntimeError("simulated mid-run crash")
        attempts["successful"] += 1
        return 0.8 if "with extra detail" in response else 0.2

    with pytest.raises(RuntimeError, match="simulated mid-run crash"):
        run_judge_audit(
            parquet_path,
            out_json,
            sample_size=6,
            judge_fn=flaky_then_crashing_judge,
            retry_base_delay=0.001,
        )

    assert attempts["count"] >= 6  # 2 retries + 4 successes before crash
    assert checkpoint_path.exists()
    lines = [line for line in checkpoint_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 4

    resumed_calls = {"count": 0}

    async def healthy_judge(prompt: str, context: str, response: str) -> float:
        resumed_calls["count"] += 1
        return 0.8 if "with extra detail" in response else 0.2

    res = run_judge_audit(
        parquet_path,
        out_json,
        sample_size=6,
        judge_fn=healthy_judge,
        retry_base_delay=0.001,
    )
    assert res["status"] == "ready"
    assert res["sample_size"] == 6
    # 6 items * 3 variants = 18 total; 4 were already checkpointed -> only 14 calls needed
    assert resumed_calls["count"] == 14


@pytest.mark.asyncio
async def test_httpx_judge_client_position_ordering_and_score_parsing():
    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(body)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"score": 0.42}'}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        judge = _create_httpx_judge_fn("deepseek", "test-key", client)
        s1 = await judge("Q?", "Ctx", "Resp")
        s2 = await judge(swap_prompt_position("Q?"), "Ctx", "Resp")

    assert s1 == pytest.approx(0.42)
    assert s2 == pytest.approx(0.42)
    user_base = captured_payloads[0]["messages"][1]["content"]
    user_swap = captured_payloads[1]["messages"][1]["content"]
    assert user_base.index("Context:") < user_base.index("Response:")
    assert user_swap.index("Response:") < user_swap.index("Context:")
    assert _parse_judge_score('```json\n{"score": 0.75}\n```') == pytest.approx(0.75)
    assert _parse_judge_score('Step 1 done. {"score": 0.35}') == pytest.approx(0.35)
    assert len(build_judge_messages("Q?", "Ctx", "Resp")) == 2
