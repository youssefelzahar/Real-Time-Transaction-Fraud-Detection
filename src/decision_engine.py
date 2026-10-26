"""Turn a fraud score into a business decision: approve, review or decline.

The model estimates risk; this module applies a configurable policy to that score.
It never loads a model or runs inference, so it can be tested and changed on its own.

Policy (see DECISION_POLICY.md):
    score <  approve_threshold                     -> approve  LOW_RISK
    approve_threshold <= score < decline_threshold -> review   REVIEW_THRESHOLD
    score >= decline_threshold                     -> decline  HIGH_RISK   (decline_mode "enforce")
                                                   -> review   HIGH_RISK   (decline_mode "shadow",
                                                                            counted as would_decline)
A score exactly at a threshold falls in the higher band.
"""
import json
import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Sequence

APPROVE = "approve"
REVIEW = "review"
DECLINE = "decline"
DECISIONS = (APPROVE, REVIEW, DECLINE)

LOW_RISK = "LOW_RISK"
REVIEW_THRESHOLD = "REVIEW_THRESHOLD"
HIGH_RISK = "HIGH_RISK"
REASON_CODES = (LOW_RISK, REVIEW_THRESHOLD, HIGH_RISK)

DECLINE_MODES = ("shadow", "enforce")

DEFAULT_POLICY_PATH = "config/decision_policy.json"


class InvalidScoreError(ValueError):
    """The score is not a finite number in [0, 1]: a technical error, not a fraud decision."""


class PolicyError(ValueError):
    """The decision policy configuration is invalid or does not fit the loaded model."""


def _finite_unit(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise PolicyError(f"{name} must be a finite number in [0, 1], got {value!r}")
    return float(value)


@dataclass(frozen=True)
class DecisionPolicy:
    policy_version: str
    approve_threshold: float
    decline_threshold: float
    decline_mode: str = "shadow"
    valid_for_models: tuple = ()
    notes: str = ""

    def __post_init__(self):
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise PolicyError("policy_version must be a non-empty string")
        approve = _finite_unit("approve_threshold", self.approve_threshold)
        decline = _finite_unit("decline_threshold", self.decline_threshold)
        if not approve < decline:
            raise PolicyError(f"need approve_threshold < decline_threshold, got {approve} and {decline}")
        if self.decline_mode not in DECLINE_MODES:
            raise PolicyError(f"decline_mode must be one of {DECLINE_MODES}, got {self.decline_mode!r}")
        object.__setattr__(self, "approve_threshold", approve)
        object.__setattr__(self, "decline_threshold", decline)
        object.__setattr__(self, "valid_for_models", tuple(self.valid_for_models))

    @classmethod
    def from_dict(cls, d):
        missing = [k for k in ("policy_version", "approve_threshold", "decline_threshold") if k not in d]
        if missing:
            raise PolicyError(f"decision policy is missing {missing}")
        return cls(policy_version=d["policy_version"], approve_threshold=d["approve_threshold"],
                   decline_threshold=d["decline_threshold"], decline_mode=d.get("decline_mode", "shadow"),
                   valid_for_models=tuple(d.get("valid_for_models", ())), notes=d.get("notes", ""))

    @classmethod
    def from_file(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def check_model(self, model_version):
        """Thresholds only mean something for the score distribution they were chosen on."""
        if self.valid_for_models and model_version not in self.valid_for_models:
            raise PolicyError(f"decision policy {self.policy_version} is not validated for model "
                              f"{model_version} (valid for: {', '.join(self.valid_for_models)})")


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    reason_code: str
    policy_version: str
    would_decline: bool = False


# A rule sees the request context and the threshold result, and returns a replacement
# result or None. Rules run in order after the thresholds. None are configured yet.
Rule = Callable[[dict, DecisionResult], Optional[DecisionResult]]


def validate_score(score):
    # numbers.Real covers int, float and NumPy scalars (XGBoost returns float32); bool is excluded.
    if isinstance(score, bool) or not isinstance(score, numbers.Real):
        raise InvalidScoreError(f"fraud score must be a number, got {type(score).__name__}")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise InvalidScoreError(f"fraud score must be a finite number in [0, 1], got {score!r}")
    return float(score)


class DecisionMetrics:
    """In-process counters for one instance. Each instance runs one event loop, so
    updates are never concurrent and need no lock."""

    LATENCY_BUCKETS_US = (5, 10, 25, 50, 100, 250, 1000)

    def __init__(self, policy):
        self.policy = policy
        self.total = 0
        self.by_decision = dict.fromkeys(DECISIONS, 0)
        self.by_reason = dict.fromkeys(REASON_CODES, 0)
        self.would_decline = 0
        self.errors = 0
        self.latency_count = 0
        self.latency_sum_us = 0.0
        self.latency_max_us = 0.0
        self.latency_buckets = [0] * (len(self.LATENCY_BUCKETS_US) + 1)

    def record(self, result, latency_us):
        self.total += 1
        self.by_decision[result.decision] += 1
        self.by_reason[result.reason_code] += 1
        self.would_decline += result.would_decline
        self.latency_count += 1
        self.latency_sum_us += latency_us
        self.latency_max_us = max(self.latency_max_us, latency_us)
        for i, bound in enumerate(self.LATENCY_BUCKETS_US):
            if latency_us <= bound:
                self.latency_buckets[i] += 1
                break
        else:
            self.latency_buckets[-1] += 1

    def record_error(self):
        self.errors += 1

    def snapshot(self):
        labels = [f"<={b}us" for b in self.LATENCY_BUCKETS_US] + [f">{self.LATENCY_BUCKETS_US[-1]}us"]
        return {
            "policy_version": self.policy.policy_version,
            "decline_mode": self.policy.decline_mode,
            "total": self.total,
            "by_decision": dict(self.by_decision),
            "by_reason": dict(self.by_reason),
            "would_decline": self.would_decline,
            "invalid_scores": self.errors,
            "decision_latency_us": {
                "count": self.latency_count,
                "mean": round(self.latency_sum_us / self.latency_count, 2) if self.latency_count else 0.0,
                "max": round(self.latency_max_us, 2),
                "buckets": dict(zip(labels, self.latency_buckets)),
            },
        }


class DecisionEngine:
    def __init__(self, policy: DecisionPolicy, rules: Sequence[Rule] = ()):
        self.policy = policy
        self.rules = tuple(rules)
        self.metrics = DecisionMetrics(policy)

    def _threshold_result(self, score):
        p = self.policy
        if score < p.approve_threshold:
            return DecisionResult(APPROVE, LOW_RISK, p.policy_version)
        if score < p.decline_threshold:
            return DecisionResult(REVIEW, REVIEW_THRESHOLD, p.policy_version)
        if p.decline_mode == "enforce":
            return DecisionResult(DECLINE, HIGH_RISK, p.policy_version)
        return DecisionResult(REVIEW, HIGH_RISK, p.policy_version, would_decline=True)

    def decide(self, fraud_probability, context=None) -> DecisionResult:
        start = perf_counter()
        try:
            score = validate_score(fraud_probability)
        except InvalidScoreError:
            self.metrics.record_error()
            raise
        result = self._threshold_result(score)
        for rule in self.rules:
            result = rule(context or {}, result) or result
        self.metrics.record(result, (perf_counter() - start) * 1e6)
        return result
