"""Data drift statistics: production feature distributions vs the training baseline (DATA_DRIFT.md).

Pure functions, no I/O. Inputs are the bounded counts the consumer already aggregates
(src/ml_monitoring.py: log bins, flags, transaction types, score histogram) and a versioned
baseline (src/monitoring_baseline.py). Reference and production use the same binning function
(src/monitoring_features.log_bin), so bins are aligned by construction.

Methods (kept to a small set on purpose):
  - PSI on <= 10 reference-mass groups (+ open groups below/above the training range), additive
    smoothing alpha=0.5 on both sides: practical effect size;
  - chi-square goodness of fit on the same groups (expected counts >= 5 after merging),
    Holm-adjusted across all tests of one window: statistical evidence;
  - binned KS distance (max CDF difference over the fine bins): descriptive only, no p-value;
  - missing-rate change (exact binomial test + absolute threshold) and unknown-category rate.
Every metric is finite or None with a reason; nothing here returns NaN or inf.
"""
import copy
import json
import math
from pathlib import Path

from scipy import stats as sps

from src.events import TRAFFIC_SOURCES
from src.ml_monitoring import KNOWN_TYPES, OTHER, SCORE_EDGES, UNKNOWN_TYPE

INSUFFICIENT = "insufficient_data"
NO_DRIFT = "no_significant_drift"
DRIFT = "drift_detected"
ERROR = "monitoring_error"
STATUS_ORDER = {ERROR: 3, DRIFT: 2, INSUFFICIENT: 1, NO_DRIFT: 0}
SEVERITY_RANK = {None: 0, "moderate": 1, "substantial": 2}

CATEGORICAL = "transaction_type"
DEFAULT_CONFIG = {
    "windows": {"1h": 60, "24h": 1440},
    "min_samples": 500,
    "psi": {"moderate": 0.10, "substantial": 0.20},
    "psi_overrides": {},                 # feature -> {"moderate": x, "substantial": y}
    "psi_smoothing_alpha": 0.5,
    "max_groups": 10,
    "alpha": 0.01,
    "min_expected": 5,
    "missing_rate_abs": 0.05,
    "unknown_rate": 0.01,
    "non_finite_rate": 0.001,
    "overflow_error_rate": 0.01,
    "critical_features": ["amount", "oldbalanceOrg", CATEGORICAL],
    "prediction_drift": True,
    "alerts": {"min_severity": "substantial", "sustained_minutes": 60, "stale_after_s": 120,
               "low_volume_min_per_hour": 100, "traffic_sources": ["production"], "expire_without_data": True},
}


class ConfigError(ValueError):
    pass


def _merge_config(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("windows", "psi_overrides"):
            out[k] = _merge_config(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    """Defaults, overridden by a JSON file (DRIFT_CONFIG / config/drift_config.json). Validated."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None and Path(path).exists():
        cfg = _merge_config(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    psi = cfg["psi"]
    if not (0 < psi["moderate"] <= psi["substantial"]):
        raise ConfigError("psi thresholds must satisfy 0 < moderate <= substantial")
    for f, o in cfg["psi_overrides"].items():
        if not (0 < o["moderate"] <= o["substantial"]):
            raise ConfigError(f"psi override for {f}: 0 < moderate <= substantial")
    if not (0 < cfg["alpha"] < 1):
        raise ConfigError("alpha must be in (0, 1)")
    if int(cfg["min_samples"]) < 1 or int(cfg["max_groups"]) < 2:
        raise ConfigError("min_samples >= 1 and max_groups >= 2")
    if not cfg["windows"] or any(int(m) < 1 for m in cfg["windows"].values()):
        raise ConfigError("windows must be {label: minutes >= 1}")
    if cfg["alerts"]["min_severity"] not in ("moderate", "substantial"):
        raise ConfigError("alerts.min_severity must be moderate or substantial")
    if not set(cfg["alerts"]["traffic_sources"]) <= set(TRAFFIC_SOURCES):
        raise ConfigError(f"alerts.traffic_sources must be a subset of {TRAFFIC_SOURCES}")
    return cfg


def thresholds_for(cfg, feature):
    return {**cfg["psi"], **cfg["psi_overrides"].get(feature, {})}


# --- small numeric helpers ----------------------------------------------------------------------

def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _r(x, nd=6):
    return round(x, nd) if _finite(x) else None


def clean_counts(raw):
    """{key: count} -> ({int bin: count}, missing, overflow, invalid). Bad counts are reported, not used."""
    bins, missing, overflow, invalid = {}, 0, 0, 0
    for k, c in (raw or {}).items():
        if not _finite(c) or c < 0:
            invalid += 1
            continue
        c = int(c)
        if k == "null":
            missing += c
        elif k == OTHER:
            overflow += c
        else:
            try:
                bins[int(k)] = bins.get(int(k), 0) + c
            except (TypeError, ValueError):
                invalid += 1
    return bins, missing, overflow, invalid


# --- grouping ------------------------------------------------------------------------------------

def make_groups(ref_counts, max_groups=10):
    """Ordered integer bins -> group upper edges by reference mass (about equal shares).

    Returns {"edges": [u1 < u2 < ...], "below": lo or None, "above": hi or None}. Group i holds
    bins (u_{i-1}, u_i]; the first group starts at the reference minimum, the last ends at the
    reference maximum. Values outside [ref_min, ref_max] fall into the open "below"/"above" groups,
    so a value never seen in training is visible instead of being absorbed by an edge group."""
    items = sorted((b, c) for b, c in ref_counts.items() if c > 0)
    if not items:
        return None
    total = sum(c for _, c in items)
    edges, cum, k = [], 0, 1
    for b, c in items:
        cum += c
        if cum >= total * k / max_groups and b != items[-1][0]:
            edges.append(b)
            while k < max_groups and cum >= total * k / max_groups:
                k += 1
    edges.append(items[-1][0])
    return {"edges": edges, "below": items[0][0], "above": items[-1][0]}


def group_index(groups, b):
    """0 = below the training range, 1..len(edges) = reference groups, len(edges)+1 = above it."""
    if b < groups["below"]:
        return 0
    if b > groups["above"]:
        return len(groups["edges"]) + 1
    for i, u in enumerate(groups["edges"]):
        if b <= u:
            return i + 1
    return len(groups["edges"]) + 1


def grouped(counts, groups):
    out = [0] * (len(groups["edges"]) + 2)
    for b, c in counts.items():
        out[group_index(groups, b)] += c
    return out


# --- statistics ----------------------------------------------------------------------------------

def psi(ref, prod, alpha=0.5):
    """Population Stability Index with additive smoothing on both sides; None if production is empty.

    p_i = (c_i + alpha) / (N + alpha*K): no zero shares, so no log(0) or division by zero.
    Under no drift its expected value is about (K-1)/N (the small-sample bias that min_samples bounds)."""
    k, n_r, n_p = len(ref), sum(ref), sum(prod)
    if k == 0 or n_p <= 0 or n_r <= 0:
        return None
    value = 0.0
    for r, p in zip(ref, prod):
        pr = (r + alpha) / (n_r + alpha * k)
        pp = (p + alpha) / (n_p + alpha * k)
        value += (pp - pr) * math.log(pp / pr)
    return value if math.isfinite(value) else None


def chi_square_gof(prod, ref, min_expected=5):
    """Production counts against reference proportions (reference treated as known).

    Adjacent groups are merged until every expected count is >= min_expected. Returns
    {"applicable": False, "reason": ...} when fewer than two groups remain."""
    n_p, n_r = sum(prod), sum(ref)
    if n_p <= 0 or n_r <= 0:
        return {"applicable": False, "reason": "no observations"}
    exp = [n_p * r / n_r for r in ref]
    m_obs, m_exp, o_acc, e_acc = [], [], 0, 0.0
    for o, e in zip(prod, exp):
        o_acc += o
        e_acc += e
        if e_acc >= min_expected:
            m_obs.append(o_acc)
            m_exp.append(e_acc)
            o_acc, e_acc = 0, 0.0
    if o_acc or e_acc:
        if m_obs:
            m_obs[-1] += o_acc
            m_exp[-1] += e_acc
        else:
            m_obs, m_exp = [o_acc], [e_acc]
    if len(m_obs) < 2:
        return {"applicable": False, "reason": f"fewer than 2 groups with expected count >= {min_expected}"}
    stat = sum((o - e) ** 2 / e for o, e in zip(m_obs, m_exp))
    df = len(m_obs) - 1
    p = float(sps.chi2.sf(stat, df))
    return {"applicable": True, "statistic": _r(stat, 4), "df": df, "p_value": p, "groups": len(m_obs)}


def ks_distance(ref_bins, prod_bins):
    """max |CDF_ref - CDF_prod| over the fine (ordered) bins. Descriptive; a lower bound of the
    KS statistic of the unbinned data. No p-value: binned, tied data violate the KS test."""
    n_r, n_p = sum(ref_bins.values()), sum(prod_bins.values())
    if n_r <= 0 or n_p <= 0:
        return None
    cr = cp = 0.0
    d = 0.0
    for b in sorted(set(ref_bins) | set(prod_bins)):
        cr += ref_bins.get(b, 0) / n_r
        cp += prod_bins.get(b, 0) / n_p
        d = max(d, abs(cr - cp))
    return d


def binomial_p(k, n, rate):
    """Two-sided exact binomial test of k events in n against `rate`; None when n == 0."""
    if n <= 0 or rate is None or not math.isfinite(rate):
        return None
    rate = min(max(rate, 0.0), 1.0)
    if rate in (0.0, 1.0):          # degenerate reference: any deviation is certain
        return 1.0 if (k == 0 and rate == 0.0) or (k == n and rate == 1.0) else 0.0
    return float(sps.binomtest(int(k), int(n), rate).pvalue)


def holm(pvalues):
    """{key: p or None} -> {key: Holm-adjusted p}; None stays None. Valid under any dependence."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None), key=lambda t: t[0])
    m, out, running = len(items), {k: None for k in pvalues}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def tvd(ref_props, prod_counts):
    n = sum(prod_counts.values())
    if n <= 0:
        return None
    keys = set(ref_props) | set(prod_counts)
    return 0.5 * sum(abs(prod_counts.get(k, 0) / n - ref_props.get(k, 0.0)) for k in keys)


# --- baseline -------------------------------------------------------------------------------------

def _counts_from_shares(shares, n):
    return {int(k): int(round(float(v) * n)) for k, v in shares.items()}


def reference(baseline):
    """Reference distributions per monitored feature, for schema 2 (counts) or schema 1 (shares)."""
    out, derived = {}, baseline.get("baseline_schema", 1) < 2
    feats = baseline["features"]
    for f in baseline.get("bins", {}):
        meta = feats.get(f, {})
        if not derived and f in baseline.get("bin_counts", {}):
            counts = {int(k): int(v) for k, v in baseline["bin_counts"][f].items()}
        else:
            n_valid = int(round(meta.get("count", 0) * (1 - (meta.get("missing_rate") or 0.0))))
            counts = _counts_from_shares(baseline["bins"][f], n_valid)
        out[f] = {"kind": "numeric", "counts": counts, "missing_rate": meta.get("missing_rate"),
                  "non_finite_rate": meta.get("non_finite_rate"), "n_total": meta.get("count")}
    for f, rate in baseline.get("flags", {}).items():
        meta = feats.get(f, {})
        if not derived and f in baseline.get("flag_counts", {}):
            fc = baseline["flag_counts"][f]
            counts = {0: int(fc.get("0", 0)), 1: int(fc.get("1", 0))}
        else:
            n_valid = int(round(meta.get("count", 0) * (1 - (meta.get("missing_rate") or 0.0))))
            counts = {1: int(round(rate * n_valid)), 0: n_valid - int(round(rate * n_valid))}
        out[f] = {"kind": "binary", "counts": counts, "missing_rate": meta.get("missing_rate"),
                  "non_finite_rate": meta.get("non_finite_rate"), "n_total": meta.get("count")}
    cat = baseline.get("categorical", {}).get(CATEGORICAL)
    if cat and not derived:
        counts, unknown_rate, missing_rate, n = dict(cat["counts"]), cat.get("unknown_rate"), cat.get(
            "missing_rate"), cat.get("n")
    else:
        n = baseline.get("split", {}).get("train_rows", 0)
        counts = {t: int(round(baseline["transaction_types"].get(t, 0.0) * n)) for t in KNOWN_TYPES}
        unknown_rate, missing_rate = 0.0, 0.0
    out[CATEGORICAL] = {"kind": "categorical", "counts": counts, "missing_rate": missing_rate,
                        "unknown_rate": unknown_rate, "n_total": n}
    for v in out.values():
        v["n_valid"] = sum(v["counts"].values())
        v["derived_counts"] = derived
    return out


def model_feature_names(models_dir, model_version):
    """Feature names of a model from models/<version>/metadata.json, or None if not on this host."""
    if models_dir is None:
        return None
    try:
        return json.loads((Path(models_dir) / model_version / "metadata.json").read_text())["feature_names"]
    except (OSError, ValueError, KeyError):
        return None


def select_baseline(baseline_dir, model_version, models_dir=None):
    """Newest baseline for the model that passes validate_baseline -> (baseline, problems, skipped).

    The single selection rule for everything that compares against a baseline (drift monitor and
    the dashboard's ML section), so they can never show different baselines. Invalid newer files
    are skipped and listed in `skipped`; `problems` explains why none was usable."""
    folder = Path(baseline_dir) / model_version
    candidates = []
    for p in folder.glob(f"bl-{model_version}-*.json"):
        suffix = p.stem.rsplit("-", 1)[-1]
        if suffix.isdigit():
            candidates.append((int(suffix), p))
    if not candidates:
        return None, ["no baseline for this model version"], []
    names = model_feature_names(models_dir, model_version)
    skipped, first_problems = [], None
    for _, p in sorted(candidates, reverse=True):
        try:
            b = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems = [f"{p.name}: unreadable ({type(e).__name__})"]
        else:
            problems = validate_baseline(b, model_version, names)
            if not problems:
                return b, [], skipped
        skipped.append({"file": p.name, "problems": problems})
        first_problems = first_problems or problems
    return None, first_problems, skipped


def validate_baseline(baseline, model_version=None, feature_names=None):
    """List of problems (empty list = usable). Never raises."""
    problems = []
    if not isinstance(baseline, dict):
        return ["baseline is not a JSON object"]
    for key in ("baseline_version", "model_version", "features", "feature_names", "bins", "flags",
                "transaction_types", "split"):
        if key not in baseline:
            problems.append(f"missing key {key!r}")
    if problems:
        return problems
    if model_version is not None and baseline["model_version"] != model_version:
        problems.append(f"baseline is for model {baseline['model_version']}, not {model_version}")
    if feature_names is not None and list(baseline["feature_names"]) != list(feature_names):
        problems.append("baseline feature names differ from the model's")
    try:
        ref = reference(baseline)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return problems + [f"reference distributions unreadable: {type(e).__name__}: {e}"]
    for f, r in ref.items():
        if any(not _finite(c) or c < 0 for c in r["counts"].values()):
            problems.append(f"{f}: non-finite or negative reference counts")
        elif r["n_valid"] <= 0:
            problems.append(f"{f}: empty reference distribution")
    for f, shares in baseline["bins"].items():
        s = sum(shares.values())
        if shares and abs(s - 1) > 1e-6:
            problems.append(f"{f}: reference shares sum to {s:.6f}, not 1")
    rs = baseline.get("reference_scores")
    if rs and list(rs.get("score_edges", [])) != list(SCORE_EDGES):
        problems.append("reference score histogram uses different edges")
    return problems


# --- evaluation -----------------------------------------------------------------------------------

def _production(stats, feature, kind):
    if kind == "numeric":
        raw = stats.get("bins", {}).get(feature, {})
    elif kind == "binary":
        raw = stats.get("flags", {}).get(feature, {})
    else:
        types = stats.get("type", {})
        known = {t: int(types.get(t, 0)) for t in KNOWN_TYPES if _finite(types.get(t, 0))}
        unknown = int(types.get(UNKNOWN_TYPE, 0) or 0)
        overflow = int(types.get(OTHER, 0) or 0)
        invalid = sum(1 for t, c in types.items() if not _finite(c) or c < 0)
        return known, 0, overflow, invalid, unknown
    bins, missing, overflow, invalid = clean_counts(raw)
    return bins, missing, overflow, invalid, 0


def evaluate_feature(feature, ref, stats, cfg):
    """One feature in one window, before multiple-comparison adjustment (see evaluate_window)."""
    kind = ref["kind"]
    th = thresholds_for(cfg, feature)
    prod, null, overflow, invalid, unknown = _production(stats, feature, kind)
    # A non-finite value (inf) is summarised as a null bin and listed in non_finite_features; it
    # is invalid input, not missing input, so it is counted and checked on its own.
    non_finite = int(stats.get("non_finite", {}).get(feature, 0) or 0) if kind != "categorical" else 0
    non_finite = min(non_finite, null)
    missing = null - non_finite
    n_valid = sum(prod.values())
    n_total = n_valid + null + overflow + unknown
    res = {"feature": feature, "type": kind, "reference_n": ref["n_valid"], "production_n": n_valid,
           "production_total": n_total, "threshold": th, "missing_rate": {
               "reference": _r(ref.get("missing_rate")), "production": _r(missing / n_total) if n_total else None},
           "non_finite_rate": {"reference": _r(ref.get("non_finite_rate")),
                               "production": _r(non_finite / n_total) if n_total else None, "n": non_finite,
                               "threshold": cfg["non_finite_rate"]} if kind != "categorical" else None,
           "psi": None, "ks_distance": None, "test": None, "missing_test": None, "reasons": [],
           "derived_reference_counts": ref["derived_counts"]}
    if kind == "categorical":
        res["unknown_rate"] = {"reference": _r(ref.get("unknown_rate")),
                               "production": _r(unknown / n_total) if n_total else None, "n": unknown,
                               "threshold": cfg["unknown_rate"]}
    if invalid:
        res["reasons"].append(f"{invalid} invalid production counts ignored")
        res["status_hint"] = ERROR
    if n_total and overflow / n_total > cfg["overflow_error_rate"]:
        res["reasons"].append(f"{overflow} observations in the aggregator's overflow bin")
        res["status_hint"] = ERROR

    # Missing-rate change (needs only the total, so it can fire when the distribution cannot).
    ref_missing = ref.get("missing_rate") or 0.0
    if n_total >= cfg["min_samples"]:
        res["missing_test"] = {"p_value": binomial_p(missing, n_total, ref_missing),
                               "abs_change": _r(missing / n_total - ref_missing)}

    if n_valid < cfg["min_samples"]:
        res["reasons"].append(f"{n_valid} valid production values < min_samples {cfg['min_samples']}")
        return res

    if kind == "numeric":
        groups = make_groups(ref["counts"], cfg["max_groups"])
        r_g, p_g = grouped(ref["counts"], groups), grouped(prod, groups)
        res["ks_distance"] = _r(ks_distance(ref["counts"], prod))
        res["groups"] = {"edges_bins": groups["edges"], "reference_share": [_r(c / sum(r_g)) for c in r_g],
                         "production_share": [_r(c / n_valid) for c in p_g],
                         "below_training_range": p_g[0], "above_training_range": p_g[-1]}
    else:
        keys = sorted(ref["counts"], key=str)
        r_g = [ref["counts"].get(k, 0) for k in keys]
        p_g = [prod.get(k, 0) for k in keys]
        res["groups"] = {"categories": [str(k) for k in keys], "reference_share": [_r(c / sum(r_g)) for c in r_g],
                         "production_share": [_r(c / n_valid) for c in p_g]}
    res["psi"] = _r(psi(r_g, p_g, cfg["psi_smoothing_alpha"]))
    res["test"] = {"name": "chi-square goodness of fit", **chi_square_gof(p_g, r_g, cfg["min_expected"])}
    return res


def _finalize(res, cfg, adjusted):
    th = res["threshold"]
    if res["test"] and res["test"].get("applicable"):
        res["test"]["p_holm"] = adjusted.get((res["feature"], "dist"))
    if res["missing_test"] is not None:
        res["missing_test"]["p_holm"] = adjusted.get((res["feature"], "missing"))
    severity = None
    if res["psi"] is not None:
        severity = "substantial" if res["psi"] >= th["substantial"] else (
            "moderate" if res["psi"] >= th["moderate"] else None)
    reasons = res["reasons"]
    dist_drift = False
    if severity is not None:
        t = res["test"] or {}
        if not t.get("applicable"):
            dist_drift = True
            reasons.append(f"PSI {res['psi']} >= {th['moderate']} (chi-square not applicable)")
        elif t["p_holm"] is not None and t["p_holm"] < cfg["alpha"]:
            dist_drift = True
            reasons.append(f"PSI {res['psi']} >= {th['moderate']} and Holm p {t['p_holm']:.2e} < {cfg['alpha']}")
        else:
            reasons.append(f"PSI {res['psi']} >= {th['moderate']} but not significant (Holm p {t['p_holm']})")
    elif res["psi"] is not None and res["test"] and res["test"].get("p_holm") is not None \
            and res["test"]["p_holm"] < cfg["alpha"]:
        reasons.append("statistically significant but PSI below threshold (large sample, small change)")
    missing_drift = False
    mt = res["missing_test"]
    if mt and mt["p_value"] is not None and abs(mt["abs_change"] or 0) >= cfg["missing_rate_abs"] \
            and mt["p_holm"] is not None and mt["p_holm"] < cfg["alpha"]:
        missing_drift = True
        reasons.append(f"missing rate changed by {mt['abs_change']:+.4f} (>= {cfg['missing_rate_abs']})")
    non_finite_drift = False
    nf = res.get("non_finite_rate")
    if nf and nf["production"] is not None and res["production_total"] >= cfg["min_samples"] \
            and nf["production"] >= cfg["non_finite_rate"]:
        non_finite_drift = True
        reasons.append(f"non-finite input rate {nf['production']} >= {cfg['non_finite_rate']} (invalid values)")
    unknown_drift = False
    ur = res.get("unknown_rate")
    if ur and ur["production"] is not None and res["production_total"] >= cfg["min_samples"] \
            and ur["production"] >= cfg["unknown_rate"]:
        unknown_drift = True
        reasons.append(f"unknown-category rate {ur['production']} >= {cfg['unknown_rate']}")

    if res.pop("status_hint", None) == ERROR:
        status = ERROR
    elif dist_drift or missing_drift or unknown_drift or non_finite_drift:
        status = DRIFT
        if missing_drift or unknown_drift or non_finite_drift:
            severity = "substantial"
    elif res["psi"] is None:
        status = INSUFFICIENT
    else:
        status = NO_DRIFT
    res["status"] = status
    res["severity"] = severity if status == DRIFT else None
    # Which checks fired: "distribution" is a population change; the others are data-quality problems.
    res["drift_kinds"] = [k for k, on in (("distribution", dist_drift), ("missing", missing_drift),
                                          ("unknown_category", unknown_drift), ("non_finite", non_finite_drift))
                          if on] if status == DRIFT else []
    return res


def prediction_drift(stats, baseline, cfg):
    """Fraud-score distribution vs the baseline's reference scores (validation split). Separate
    from input drift: it can move because inputs moved, or because the model's behaviour did."""
    rs = baseline.get("reference_scores")
    out = {"reference": "validation split scores (baseline reference_scores)", "status": INSUFFICIENT,
           "psi": None, "test": None, "reasons": []}
    if not rs:
        out["reasons"].append("baseline has no reference score histogram")
        return out
    ref_hist = {i: int(c) for i, c in enumerate(rs["score_hist"])}
    prod, _, overflow, invalid = clean_counts(stats.get("score_hist", {}))
    n = sum(prod.values())
    out.update({"reference_n": sum(ref_hist.values()), "production_n": n})
    if invalid or overflow:
        out["status"] = ERROR
        out["reasons"].append("invalid or overflow score-histogram counts")
        return out
    if n < cfg["min_samples"]:
        out["reasons"].append(f"{n} predictions < min_samples {cfg['min_samples']}")
        return out
    groups = make_groups(ref_hist, cfg["max_groups"])
    r_g, p_g = grouped(ref_hist, groups), grouped(prod, groups)
    out["psi"] = _r(psi(r_g, p_g, cfg["psi_smoothing_alpha"]))
    t = chi_square_gof(p_g, r_g, cfg["min_expected"])
    out["test"] = {"name": "chi-square goodness of fit", **t}
    th = cfg["psi"]
    big = out["psi"] is not None and out["psi"] >= th["moderate"]
    sig = (not t.get("applicable")) or t["p_value"] < cfg["alpha"]
    out["status"] = DRIFT if big and sig else NO_DRIFT
    out["severity"] = ("substantial" if out["psi"] >= th["substantial"] else "moderate") if out["status"] == DRIFT \
        else None
    out["groups"] = {"edges_score": [SCORE_EDGES[u + 1] for u in groups["edges"]],
                     "reference_share": [_r(c / sum(r_g)) for c in r_g],
                     "production_share": [_r(c / n) for c in p_g]}
    return out


def evaluate_window(stats, baseline, cfg, *, baseline_problems=()):
    """All monitored features for one model version in one window: results + summary."""
    if baseline is None or baseline_problems:
        reason = "no baseline for this model version" if baseline is None else "; ".join(baseline_problems)
        return {"features": [], "prediction": None, "summary": {
            "status": INSUFFICIENT, "reason": reason, "production_n": stats.get("n", 0)}}
    ref = reference(baseline)
    results = [evaluate_feature(f, r, stats, cfg) for f, r in ref.items()]
    pvals = {}
    for r in results:
        if r["test"] and r["test"].get("applicable"):
            pvals[(r["feature"], "dist")] = r["test"]["p_value"]
        if r["missing_test"] is not None:
            pvals[(r["feature"], "missing")] = r["missing_test"]["p_value"]
    adjusted = holm(pvals)
    results = [_finalize(r, cfg, adjusted) for r in results]
    for r in results:
        r["critical"] = r["feature"] in cfg["critical_features"]
        if r["test"] and "p_value" in r["test"]:
            r["test"]["p_value"] = _r(r["test"]["p_value"], 12)
        if r["missing_test"] and r["missing_test"]["p_value"] is not None:
            r["missing_test"]["p_value"] = _r(r["missing_test"]["p_value"], 12)
    results.sort(key=lambda r: (-STATUS_ORDER[r["status"]], -(r["psi"] or 0)))

    counts = {s: sum(r["status"] == s for r in results) for s in STATUS_ORDER}
    sufficient = [r for r in results if r["status"] in (DRIFT, NO_DRIFT)]
    ranked = sorted((r for r in results if r["psi"] is not None), key=lambda r: -r["psi"])
    type_counts, _, _, _, _ = _production(stats, CATEGORICAL, "categorical")
    ref_type = ref[CATEGORICAL]
    ref_props = {k: v / ref_type["n_valid"] for k, v in ref_type["counts"].items()} if ref_type["n_valid"] else {}
    if counts[ERROR]:
        overall = ERROR
    elif counts[DRIFT]:
        overall = DRIFT
    elif not sufficient:
        overall = INSUFFICIENT
    else:
        overall = NO_DRIFT
    summary = {
        "status": overall,
        "features_monitored": len(results),
        "features_with_sufficient_data": len(sufficient),
        "features_drifting": counts[DRIFT],
        "features_insufficient": counts[INSUFFICIENT],
        "features_error": counts[ERROR],
        "share_of_features_drifting": _r(counts[DRIFT] / len(sufficient)) if sufficient else None,
        "largest_changes": [{"feature": r["feature"], "psi": r["psi"], "status": r["status"]} for r in ranked[:3]],
        "production_n": stats.get("n", 0),
        "feature_summary_available": stats.get("summary_available", 0),
        # Like every other statistic, not reported on fewer than min_samples observations.
        "type_mix_tvd": _r(tvd(ref_props, type_counts)) if sum(type_counts.values()) >= cfg["min_samples"] else None,
        "note": "No combined drift score: features are dependent and these are marginal checks.",
    }
    pred = prediction_drift(stats, baseline, cfg) if cfg.get("prediction_drift", True) else None
    return {"features": results, "prediction": pred, "summary": summary}
