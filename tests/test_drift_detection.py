"""Data drift statistics (src/drift_detection.py) on synthetic, deterministic distributions."""
import json
import math

import pytest

from src import drift_detection as dd
from src.ml_monitoring import OTHER, SCORE_EDGES, UNKNOWN_TYPE

# Reference: training counts (schema 2). amount ~ log bins 12..22, orig_bal_zero 30% ones,
# transaction types like PaySim, scores mostly near zero.
AMOUNT = {12: 50_000, 13: 100_000, 14: 150_000, 15: 200_000, 16: 200_000, 17: 150_000, 18: 80_000,
          19: 40_000, 20: 20_000, 21: 8_000, 22: 2_000}
FLAG = {0: 700_000, 1: 300_000}
TYPES = {"CASH_IN": 220_000, "CASH_OUT": 355_000, "DEBIT": 6_000, "PAYMENT": 337_000, "TRANSFER": 82_000}
SCORES = [600_000, 200_000, 100_000, 50_000, 30_000, 10_000, 5_000, 2_000, 1_000, 800, 600, 400, 300, 200,
          150, 100, 80, 60, 50, 40, 30, 30, 20, 20, 15, 10, 10, 5, 5, 5, 5, 0]
N_REF = 1_000_000


def baseline(model="m1", schema=2):
    b = {
        "baseline_version": f"bl-{model}-1", "model_version": model, "baseline_schema": schema,
        "feature_names": ["amount", "orig_bal_zero", "type_code"],
        "features": {"amount": {"count": N_REF, "missing_rate": 0.0},
                     "orig_bal_zero": {"count": N_REF, "missing_rate": 0.0},
                     "type_code": {"count": N_REF, "missing_rate": 0.0}},
        "bins": {"amount": {str(k): v / N_REF for k, v in AMOUNT.items()}},
        "flags": {"orig_bal_zero": FLAG[1] / N_REF},
        "transaction_types": {t: c / N_REF for t, c in TYPES.items()},
        "split": {"train_rows": N_REF},
        "reference_scores": {"score_edges": list(SCORE_EDGES), "score_hist": SCORES, "n": sum(SCORES)},
    }
    if schema >= 2:
        b.update({"bin_counts": {"amount": {str(k): v for k, v in AMOUNT.items()}},
                  "flag_counts": {"orig_bal_zero": {"0": FLAG[0], "1": FLAG[1], "null": 0}},
                  "categorical": {"transaction_type": {"n": N_REF, "counts": dict(TYPES), "missing_rate": 0.0,
                                                       "unknown_rate": 0.0}},
                  "feature_pipeline": {"id": "T-1"}})
    return b


def scaled(counts, n):
    """Exact proportions scaled to n (largest remainder), so 'identical' is really identical."""
    total = sum(counts.values())
    raw = {k: c * n / total for k, c in counts.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def stats(n=5000, amount=None, flag=None, types=None, scores=None, missing=0, unknown=0):
    amount = amount if amount is not None else scaled(AMOUNT, n - missing)
    flag = flag if flag is not None else scaled(FLAG, n)
    types = types if types is not None else scaled(TYPES, n - unknown)
    s = {"n": n, "summary_available": n,
         "bins": {"amount": {**{str(k): v for k, v in amount.items()}, **({"null": missing} if missing else {})}},
         "flags": {"orig_bal_zero": {str(k): v for k, v in flag.items()}},
         "type": {**types, **({UNKNOWN_TYPE: unknown} if unknown else {})}}
    hist = scores if scores is not None else scaled(dict(enumerate(SCORES)), n)
    s["score_hist"] = {str(k): v for k, v in hist.items()}
    return s


CFG = dd.load_config(None)


def by_feature(out):
    return {r["feature"]: r for r in out["features"]}


# --- no drift / drift ------------------------------------------------------------------------------

def test_identical_distributions_no_drift():
    out = dd.evaluate_window(stats(), baseline(), CFG)
    for r in out["features"]:
        assert r["status"] == dd.NO_DRIFT, r
        assert r["psi"] < 0.01 and math.isfinite(r["psi"])
    assert out["summary"]["status"] == dd.NO_DRIFT and out["summary"]["features_drifting"] == 0
    assert out["prediction"]["status"] == dd.NO_DRIFT


def test_numeric_shift_amount_times_ten():
    shifted = {k + 4: v for k, v in scaled(AMOUNT, 5000).items()}      # +4 quarter-decades = x10
    r = by_feature(dd.evaluate_window(stats(amount=shifted), baseline(), CFG))["amount"]
    assert r["status"] == dd.DRIFT and r["severity"] == "substantial"
    assert r["psi"] >= 0.2 and r["test"]["p_holm"] < 0.01 and r["ks_distance"] > 0.5
    assert r["groups"]["above_training_range"] > 0          # values never seen in training are visible


def test_categorical_shift():
    t = scaled({"CASH_IN": 1, "CASH_OUT": 1, "DEBIT": 1, "PAYMENT": 1, "TRANSFER": 6}, 5000)
    out = dd.evaluate_window(stats(types=t), baseline(), CFG)
    r = by_feature(out)["transaction_type"]
    assert r["status"] == dd.DRIFT and r["psi"] >= 0.2
    assert out["summary"]["type_mix_tvd"] > 0.3


def test_binary_flag_shift():
    r = by_feature(dd.evaluate_window(stats(flag={0: 2000, 1: 3000}), baseline(), CFG))["orig_bal_zero"]
    assert r["status"] == dd.DRIFT and r["type"] == "binary"


def test_missing_rate_change_detected_even_without_distribution_change():
    r = by_feature(dd.evaluate_window(stats(missing=1000), baseline(), CFG))["amount"]
    assert r["status"] == dd.DRIFT and r["missing_rate"]["production"] == 0.2
    assert r["psi"] < 0.01                                            # valid values unchanged
    assert any("missing rate" in x for x in r["reasons"])


def test_unknown_categories():
    r = by_feature(dd.evaluate_window(stats(unknown=250), baseline(), CFG))["transaction_type"]
    assert r["status"] == dd.DRIFT and r["unknown_rate"]["production"] == 0.05
    assert r["psi"] < 0.01                     # unknowns are excluded from PSI, reported on their own


# --- insufficient / empty / invalid ---------------------------------------------------------------

def test_empty_production_is_insufficient_never_healthy():
    empty = {"n": 0, "summary_available": 0, "bins": {}, "flags": {}, "type": {}, "score_hist": {}}
    out = dd.evaluate_window(empty, baseline(), CFG)
    assert {r["status"] for r in out["features"]} == {dd.INSUFFICIENT}
    assert all(r["psi"] is None for r in out["features"])
    assert out["summary"]["status"] == dd.INSUFFICIENT and out["prediction"]["status"] == dd.INSUFFICIENT


def test_below_min_samples_is_insufficient():
    out = dd.evaluate_window(stats(n=499), baseline(), CFG)
    assert {r["status"] for r in out["features"]} == {dd.INSUFFICIENT}
    assert all("min_samples" in r["reasons"][0] for r in out["features"])


def test_invalid_and_non_finite_counts_are_monitoring_errors():
    s = stats()
    s["bins"]["amount"]["15"] = float("nan")
    s["bins"]["amount"]["16"] = -3
    s["flags"]["orig_bal_zero"]["1"] = float("inf")
    f = by_feature(dd.evaluate_window(s, baseline(), CFG))
    assert f["amount"]["status"] == dd.ERROR and f["orig_bal_zero"]["status"] == dd.ERROR
    assert all(r["psi"] is None or math.isfinite(r["psi"]) for r in f.values())


def test_aggregator_overflow_bin_is_an_error():
    s = stats()
    s["bins"]["amount"][OTHER] = 200          # 4% of observations could not be binned
    assert by_feature(dd.evaluate_window(s, baseline(), CFG))["amount"]["status"] == dd.ERROR


def test_no_baseline_or_invalid_baseline():
    out = dd.evaluate_window(stats(), None, CFG)
    assert out["summary"]["status"] == dd.INSUFFICIENT and "no baseline" in out["summary"]["reason"]
    out = dd.evaluate_window(stats(), baseline(), CFG, baseline_problems=["wrong model"])
    assert out["summary"]["status"] == dd.INSUFFICIENT and out["features"] == []


# --- statistics edge cases ------------------------------------------------------------------------

def test_psi_zero_bins_and_empty():
    assert dd.psi([100, 0, 0], [0, 0, 100]) > 1 and math.isfinite(dd.psi([100, 0, 0], [0, 0, 100]))
    assert dd.psi([0, 0], [0, 0]) is None and dd.psi([10, 10], [0, 0]) is None
    assert dd.psi([50, 50], [50, 50]) == pytest.approx(0.0, abs=1e-12)


def test_chi_square_not_applicable_and_merging():
    t = dd.chi_square_gof([3, 0], [1, 1_000_000])
    assert t["applicable"] is False and "fewer than 2" in t["reason"]
    t = dd.chi_square_gof([0, 0], [1, 1])
    assert t["applicable"] is False
    t = dd.chi_square_gof([100, 0, 0, 900], [10, 1, 1, 988])       # expected 1, 1 merged into a neighbour
    assert t["applicable"] is True and t["groups"] == 2 and 0 <= t["p_value"] <= 1


def test_large_sample_tiny_change_is_significant_but_not_drift():
    big = 2_000_000
    s = stats(n=big, flag={0: int(big * 0.705), 1: big - int(big * 0.705)})
    r = by_feature(dd.evaluate_window(s, baseline(), CFG))["orig_bal_zero"]
    assert r["test"]["p_holm"] < 1e-6 and r["psi"] < 0.01
    assert r["status"] == dd.NO_DRIFT
    assert any("statistically significant but PSI below threshold" in x for x in r["reasons"])


def test_holm_adjustment():
    adj = dd.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06), "d": None}
    assert dd.holm({}) == {}


def test_ks_distance_and_binomial_edges():
    assert dd.ks_distance({1: 50, 2: 50}, {1: 50, 2: 50}) == 0
    assert dd.ks_distance({1: 100}, {2: 100}) == 1
    assert dd.ks_distance({}, {1: 1}) is None
    assert dd.binomial_p(0, 100, 0.0) == 1.0 and dd.binomial_p(5, 100, 0.0) == 0.0
    assert dd.binomial_p(0, 0, 0.5) is None and 0 < dd.binomial_p(50, 100, 0.5) <= 1


def test_groups_cover_every_bin():
    g = dd.make_groups(AMOUNT, 10)
    assert len(g["edges"]) <= 10 and g["edges"][-1] == 22
    assert dd.group_index(g, 5) == 0 and dd.group_index(g, 40) == len(g["edges"]) + 1
    assert sum(dd.grouped(AMOUNT, g)) == sum(AMOUNT.values())
    assert dd.make_groups({}, 10) is None


def test_prediction_drift_reported_separately():
    hist = scaled({i: 1 for i in range(20, 31)}, 3000)        # every score high
    out = dd.evaluate_window(stats(n=3000, scores=hist), baseline(), CFG)
    assert out["prediction"]["status"] == dd.DRIFT
    assert out["summary"]["features_drifting"] == 0           # inputs unchanged: not input drift


# --- baseline, versions, config -------------------------------------------------------------------

def test_schema_1_baseline_gives_the_same_result_with_derived_counts():
    s = stats(amount={k + 4: v for k, v in scaled(AMOUNT, 5000).items()})
    r1 = by_feature(dd.evaluate_window(s, baseline(schema=1), CFG))["amount"]
    r2 = by_feature(dd.evaluate_window(s, baseline(schema=2), CFG))["amount"]
    assert r1["derived_reference_counts"] is True and r2["derived_reference_counts"] is False
    assert r1["psi"] == pytest.approx(r2["psi"], rel=1e-3) and r1["status"] == r2["status"]


def test_validate_baseline_versions_and_corruption():
    b = baseline("m1")
    assert dd.validate_baseline(b, "m1", ["amount", "orig_bal_zero", "type_code"]) == []
    assert "not m2" in dd.validate_baseline(b, "m2")[0]
    assert "feature names differ" in dd.validate_baseline(b, "m1", ["amount"])[0]
    bad = json.loads(json.dumps(b))
    bad["bin_counts"]["amount"]["15"] = -1
    assert any("negative" in p for p in dd.validate_baseline(bad, "m1"))
    bad = json.loads(json.dumps(b))
    bad["reference_scores"]["score_edges"] = [0, 1]
    assert any("edges" in p for p in dd.validate_baseline(bad, "m1"))
    assert dd.validate_baseline({"model_version": "m1"})[0].startswith("missing key")
    assert dd.validate_baseline([]) == ["baseline is not a JSON object"]


def test_config_file_overrides_and_validation(tmp_path):
    p = tmp_path / "drift.json"
    p.write_text(json.dumps({"min_samples": 100, "psi_overrides": {"amount": {"moderate": 0.5, "substantial": 0.9}}}))
    cfg = dd.load_config(p)
    assert cfg["min_samples"] == 100 and cfg["psi"]["moderate"] == 0.10
    assert dd.thresholds_for(cfg, "amount") == {"moderate": 0.5, "substantial": 0.9}
    shifted = {k + 2: v for k, v in scaled(AMOUNT, 5000).items()}
    r = by_feature(dd.evaluate_window(stats(amount=shifted), baseline(), cfg))["amount"]
    assert r["threshold"]["moderate"] == 0.5
    p.write_text(json.dumps({"psi": {"moderate": 0.3, "substantial": 0.2}}))
    with pytest.raises(dd.ConfigError):
        dd.load_config(p)
    assert dd.load_config(tmp_path / "missing.json")["min_samples"] == 500
