import json

import numpy as np
import pytest

from rethink_mh.experiments.audit_fit import decide_from_files, fit_from_files
from rethink_mh.rethinking.audit import (
    AuditEstimator,
    audit_feature_names,
    fit_audit,
    fit_logistic_l2,
    fit_source_validity,
    roc_auc,
    select_risk_threshold,
)
from rethink_mh.rethinking.trigger import RethinkPolicy, validity_weighted_disagreement


def _assessment(risk, confidence, audio, visual, reliability=(0.9, 0.9)):
    return {
        "risk_probability": risk,
        "confidence": confidence,
        "audio_source": {
            "status": "available",
            "risk_probability": audio,
            "reliability": reliability[0],
        },
        "visual_source": {
            "status": "available",
            "risk_probability": visual,
            "reliability": reliability[1],
        },
        "supporting_segment_ids": ["S001"],
        "contradictory_segment_ids": [],
        "uncertain_segment_ids": [],
        "requested_segment_ids": ["S002"],
    }


class _Duck:
    """Minimal assessment with the attributes the trigger reads."""

    def __init__(self, data):
        self.risk_probability = data["risk_probability"]
        self.confidence = data["confidence"]
        self.source_risk_probabilities = {
            "audio": data["audio_source"]["risk_probability"],
            "visual": data["visual_source"]["risk_probability"],
        }
        self.source_reliability = {
            "audio": data["audio_source"]["reliability"],
            "visual": data["visual_source"]["reliability"],
        }
        self.supporting_segment_ids = tuple(data["supporting_segment_ids"])
        self.contradictory_segment_ids = ()
        self.uncertain_segment_ids = ()
        self.requested_segment_ids = tuple(data["requested_segment_ids"])


def test_roc_auc_handles_ties_and_rejects_one_class():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.2, 0.3])


def test_source_validity_is_normalized_auroc_and_skips_unavailable():
    labels = [0, 0, 1, 1, 1]
    validity = fit_source_validity(
        labels,
        {"audio": [0.1, 0.2, 0.8, 0.9, None], "visual": [0.9, 0.8, 0.2, 0.1, 0.5]},
    )
    assert validity == {"audio": 1.0, "visual": 0.0}


def test_disagreement_matches_paper_definition():
    risks = {"audio": 0.8, "visual": 0.2}
    reliability = {"audio": 0.9, "visual": 0.5}
    d_rel, d_unrel, pairs = validity_weighted_disagreement(
        risks, reliability, {"audio": 0.5, "visual": 1.0}, reliable_source_threshold=0.6
    )
    assert pairs == 1
    assert d_rel == pytest.approx(0.9 * 0.5 * 0.5 * 1.0 * 0.6)
    assert d_unrel == pytest.approx((1.0 - 0.5) * 0.6)


def test_logistic_fit_separates_and_penalty_shrinks():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] + 0.2 * rng.normal(size=400) > 0).astype(float)
    weak, _ = fit_logistic_l2(x, y, l2=100.0)
    strong, _ = fit_logistic_l2(x, y, l2=0.1)
    assert strong[0] > 2.0 and abs(strong[1]) < abs(strong[0]) / 4
    assert abs(weak[0]) < abs(strong[0])


def test_threshold_hits_target_rate_without_labels():
    risks = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    threshold = select_risk_threshold(risks, 0.3)
    assert sum(risk >= threshold for risk in risks) == 3
    assert select_risk_threshold(risks, 0.0) == 1.0


def _fitted_estimator():
    rows, targets = [], []
    rng = np.random.default_rng(1)
    for _ in range(120):
        wrong = rng.random() < 0.35
        gap = rng.uniform(0.4, 0.7) if wrong else rng.uniform(0.0, 0.2)
        risk = rng.uniform(0.3, 0.7)
        rows.append(
            _Duck(_assessment(risk, rng.uniform(0.4, 0.9), risk + gap / 2, risk - gap / 2))
        )
        targets.append(int(wrong))
    return fit_audit(rows, targets, source_validity={"audio": 0.8, "visual": 0.6}), rows


def test_fit_audit_ranks_disagreement_as_risk_and_round_trips(tmp_path):
    estimator, rows = _fitted_estimator()
    assert estimator.feature_names == audit_feature_names()
    index = estimator.feature_names.index("reliable_disagreement")
    assert estimator.coefficients[index] > 0
    estimator = estimator.with_threshold(0.5)
    path = tmp_path / "audit.json"
    estimator.save(path)
    loaded = AuditEstimator.load(path)
    assert loaded.assessment_risk(rows[0]) == pytest.approx(estimator.assessment_risk(rows[0]))


def test_policy_triggers_on_audit_risk_and_uses_fitted_validity():
    estimator, _ = _fitted_estimator()
    estimator = estimator.with_threshold(0.5)
    policy = RethinkPolicy(audit=estimator)
    conflict = policy.evaluate(_Duck(_assessment(0.5, 0.9, 0.95, 0.05)), ["S001", "S002"])
    agree = policy.evaluate(_Duck(_assessment(0.5, 0.9, 0.52, 0.48)), ["S001", "S002"])
    assert conflict.should_rethink and not agree.should_rethink
    assert conflict.trigger_score == pytest.approx(conflict.metrics["audit_risk"])
    assert conflict.selected_segment_ids[0] == "S002"
    expected = 0.9 * 0.9 * 0.8 * 0.6 * 0.9
    assert conflict.metrics["weighted_reliable_disagreement"] == pytest.approx(expected)


def test_fit_and_decide_from_frozen_files(tmp_path):
    rng = np.random.default_rng(2)
    raw, outcomes, dev = [], [], []
    for index in range(80):
        label = int(rng.random() < 0.4)
        wrong = rng.random() < 0.3
        gap = rng.uniform(0.4, 0.7) if wrong else rng.uniform(0.0, 0.2)
        probability = (0.7 if label else 0.3) if not wrong else (0.3 if label else 0.7)
        audio = min(0.99, probability + gap / 2)
        visual = max(0.01, probability - gap / 2) if label else max(0.01, 0.2 - gap / 4)
        record = _assessment(probability, 0.7, audio, visual)
        raw.append({"session_id": f"s{index}", "status": "ok", "initial": {"initial_assessment": record}})
        outcomes.append(
            {"session_id": f"s{index}", "label": label, "initial_probability": probability,
             "initial_threshold": 0.5}
        )
        dev.append({"session_id": f"d{index}", "status": "ok", "initial": {"initial_assessment": record}})
    raw.append({"session_id": "failed", "status": "initial_failed"})
    outcomes.append({"session_id": "failed", "label": 1, "initial_probability": 0.4,
                     "initial_threshold": 0.5})
    for name, rows in (("raw.jsonl", raw), ("outcomes.jsonl", outcomes), ("dev.jsonl", dev)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    estimator = fit_from_files(
        raw_path=tmp_path / "raw.jsonl",
        outcomes_path=tmp_path / "outcomes.jsonl",
        development_raw_path=tmp_path / "dev.jsonl",
        target_trigger_rate=0.25,
        l2=1.0,
    )
    assert estimator.metadata["fit_sessions"] == 80
    estimator.save(tmp_path / "audit.json")
    decisions = decide_from_files(audit_path=tmp_path / "audit.json", raw_path=tmp_path / "raw.jsonl")
    by_id = {row["session_id"]: row for row in decisions}
    assert by_id["failed"]["should_rethink"] is True
    rate = np.mean([row["should_rethink"] for row in decisions if row["status"] == "ok"])
    assert rate == pytest.approx(0.25, abs=0.05)
