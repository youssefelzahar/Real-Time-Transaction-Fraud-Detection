import json
import math

import pytest

from src.decision_engine import (APPROVE, DECLINE, HIGH_RISK, LOW_RISK, REVIEW, REVIEW_THRESHOLD,
                                 DecisionEngine, DecisionPolicy, DecisionResult, InvalidScoreError,
                                 PolicyError)

A, D = 0.54, 0.999


def engine(mode="shadow", rules=()):
    return DecisionEngine(DecisionPolicy("dp-test", A, D, decline_mode=mode, valid_for_models=("e7_v2",)),
                          rules=rules)


# --- boundaries: a score exactly at a threshold falls in the higher band ---------------

@pytest.mark.parametrize("mode", ["shadow", "enforce"])
@pytest.mark.parametrize("score, decision, reason", [
    (0.0, APPROVE, LOW_RISK),
    (0.5399, APPROVE, LOW_RISK),          # below approve
    (A, REVIEW, REVIEW_THRESHOLD),        # exactly at approve
    (0.8, REVIEW, REVIEW_THRESHOLD),      # between
    (0.9989, REVIEW, REVIEW_THRESHOLD),   # just below decline
])
def test_bands_below_decline_are_the_same_in_both_modes(mode, score, decision, reason):
    r = engine(mode).decide(score)
    assert (r.decision, r.reason_code, r.would_decline) == (decision, reason, False)


@pytest.mark.parametrize("score", [D, 0.9995, 1.0])   # exactly at decline, above, maximum
def test_high_risk_enforce_declines(score):
    r = engine("enforce").decide(score)
    assert (r.decision, r.reason_code, r.would_decline) == (DECLINE, HIGH_RISK, False)


@pytest.mark.parametrize("score", [D, 0.9995, 1.0])
def test_high_risk_shadow_reviews_but_flags_would_decline(score):
    r = engine("shadow").decide(score)
    assert (r.decision, r.reason_code, r.would_decline) == (REVIEW, HIGH_RISK, True)


def test_result_carries_policy_version_and_is_deterministic():
    e = engine()
    assert {e.decide(0.7) for _ in range(100)} == {DecisionResult(REVIEW, REVIEW_THRESHOLD, "dp-test")}


# --- invalid scores are technical errors, never decisions -----------------------------

@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf, -math.inf, None, "0.5", True, [0.5]])
def test_invalid_scores_raise(bad):
    e = engine()
    with pytest.raises(InvalidScoreError):
        e.decide(bad)
    assert e.metrics.total == 0 and e.metrics.errors == 1


def test_numpy_floats_are_accepted():
    np = pytest.importorskip("numpy")
    assert engine().decide(np.float32(0.1)).decision == APPROVE


# --- policy configuration --------------------------------------------------------------

@pytest.mark.parametrize("approve, decline", [
    (0.9, 0.5), (0.5, 0.5),             # approve must be below decline
    (-0.1, 0.5), (0.5, 1.1),            # outside [0, 1]
    (math.nan, 0.9), (0.5, math.inf),   # not finite
    ("0.5", 0.9), (True, 0.9),          # not numbers
])
def test_invalid_thresholds_rejected(approve, decline):
    with pytest.raises(PolicyError):
        DecisionPolicy("dp-x", approve, decline)


@pytest.mark.parametrize("kwargs", [{"decline_mode": "block"}, {"policy_version": ""}, {"policy_version": "  "}])
def test_invalid_mode_or_version_rejected(kwargs):
    args = {"policy_version": "dp-x", "approve_threshold": 0.5, "decline_threshold": 0.9, **kwargs}
    with pytest.raises(PolicyError):
        DecisionPolicy(**args)


def test_edge_thresholds_allowed():
    DecisionPolicy("dp-x", 0.0, 1.0)


def test_from_dict_requires_thresholds_and_version():
    with pytest.raises(PolicyError, match="missing"):
        DecisionPolicy.from_dict({"policy_version": "dp-x", "approve_threshold": 0.5})


def test_repository_policy_file_is_valid():
    p = DecisionPolicy.from_file("config/decision_policy.json")
    assert p.approve_threshold < p.decline_threshold
    assert p.decline_mode in ("shadow", "enforce")
    assert "e7_v2" in p.valid_for_models


def test_from_file_roundtrip(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"policy_version": "dp-f", "approve_threshold": 0.3,
                                "decline_threshold": 0.95, "decline_mode": "enforce",
                                "valid_for_models": ["m1"]}))
    p = DecisionPolicy.from_file(path)
    assert (p.approve_threshold, p.decline_threshold, p.decline_mode, p.valid_for_models) == \
        (0.3, 0.95, "enforce", ("m1",))


def test_policy_refuses_unvalidated_model():
    p = DecisionPolicy("dp-x", 0.5, 0.9, valid_for_models=("e7_v1", "e7_v2"))
    p.check_model("e7_v2")
    with pytest.raises(PolicyError, match="not validated"):
        p.check_model("e8_v1")


# --- reason codes, rules, metrics -----------------------------------------------------

def test_reason_codes_match_decisions():
    allowed = {LOW_RISK: {APPROVE}, REVIEW_THRESHOLD: {REVIEW}, HIGH_RISK: {REVIEW, DECLINE}}
    for mode in ("shadow", "enforce"):
        e = engine(mode)
        for i in range(1001):
            r = e.decide(i / 1000)
            assert r.decision in allowed[r.reason_code]


def test_rules_run_after_thresholds_and_can_override():
    seen = []

    def rule(context, result):
        seen.append((context, result.decision))
        if context.get("force_review") and result.decision == APPROVE:
            return DecisionResult(REVIEW, REVIEW_THRESHOLD, result.policy_version)
        return None

    e = engine(rules=[rule])
    assert e.decide(0.1).decision == APPROVE
    assert e.decide(0.1, {"force_review": True}).decision == REVIEW
    assert seen == [({}, APPROVE), ({"force_review": True}, APPROVE)]


def test_metrics_count_decisions_reasons_and_latency():
    e = engine("shadow")
    for s in (0.1, 0.2, 0.6, 0.9995):
        e.decide(s)
    m = e.metrics.snapshot()
    assert m["total"] == 4
    assert m["by_decision"] == {APPROVE: 2, REVIEW: 2, DECLINE: 0}
    assert m["by_reason"] == {LOW_RISK: 2, REVIEW_THRESHOLD: 1, HIGH_RISK: 1}
    assert m["would_decline"] == 1
    assert m["policy_version"] == "dp-test" and m["decline_mode"] == "shadow"
    lat = m["decision_latency_us"]
    assert lat["count"] == 4 and sum(lat["buckets"].values()) == 4 and lat["max"] >= lat["mean"] > 0
