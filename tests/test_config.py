from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_configs_have_required_keys():
    rag = yaml.safe_load((ROOT_DIR / "config/ragtruth_conformal.yaml").read_text())
    rtr = yaml.safe_load((ROOT_DIR / "config/routerbench.yaml").read_text())
    assert rag["random_state"] == 42
    assert 0.0 < rag["default_alpha"] < 0.5
    assert rtr["ood_holdout_tasks"]
    assert 0.0 <= rtr["default_lambda_cost"] <= 1.0
