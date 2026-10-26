"""Evidence for decision-policy thresholds, from the VALIDATION period only.

    python -m src.decision_policy_analysis --model-dir models/e7_v2
    python -m src.decision_policy_analysis --model-dir models/e7_v2 --policy config/decision_policy.json

Prints, for the model's validation split (same chronological split as training):
  - what the configured approve threshold does,
  - decline-threshold candidates: declined, legitimate declined, fraud declined,
    review volume left, fraud left in review,
  - calibration: observed fraud rate per score band (scores are NOT calibrated
    probabilities: training reweights fraud with scale_pos_weight).
The test split is never scored here; thresholds must not be tuned on it.
"""
import argparse

import numpy as np

from src.data_preprocessing import TARGET, chronological_split, load_data
from src.decision_engine import DEFAULT_POLICY_PATH, DecisionPolicy
from src.predict import FraudPredictor

DECLINE_CANDIDATES = (0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
SCORE_BANDS = (0, 0.01, 0.1, 0.3, 0.54, 0.7, 0.9, 0.99, 0.999, 1.0)


def validation_scores(predictor, data_path):
    df = load_data(data_path)
    _, val, test, bounds = chronological_split(df)
    assert val["step"].max() < test["step"].min(), "validation must precede the test period"
    X = predictor.pipeline.transform(val)
    scores = predictor.booster.inplace_predict(X.to_numpy())
    return scores, val[TARGET].to_numpy().astype(bool), bounds["val"]


def decline_table(scores, y, approve):
    rows = []
    for t in DECLINE_CANDIDATES:
        if t <= approve:
            continue
        declined = scores >= t
        review = (scores >= approve) & ~declined
        rows.append({"decline_threshold": t, "declined": int(declined.sum()),
                     "legit_declined": int((declined & ~y).sum()), "fraud_declined": int((declined & y).sum()),
                     "review_volume": int(review.sum()), "fraud_in_review": int((review & y).sum()),
                     "decline_precision": float((declined & y).sum() / max(declined.sum(), 1))})
    return rows


def calibration(scores, y):
    out = []
    for lo, hi in zip(SCORE_BANDS[:-1], SCORE_BANDS[1:]):
        band = (scores >= lo) & ((scores < hi) if hi < 1.0 else (scores <= hi))
        if band.any():
            out.append({"band": f"[{lo}, {hi}{']' if hi == 1.0 else ')'}", "n": int(band.sum()),
                        "mean_score": float(scores[band].mean()), "fraud_rate": float(y[band].mean())})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models/e7_v2")
    ap.add_argument("--data", default="data/fraud_data.csv")
    ap.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    args = ap.parse_args()

    predictor = FraudPredictor.load(args.model_dir)
    policy = DecisionPolicy.from_file(args.policy)
    scores, y, steps = validation_scores(predictor, args.data)
    approve, decline = policy.approve_threshold, policy.decline_threshold

    print(f"model {predictor.model_version} | policy {policy.policy_version} ({policy.decline_mode}) | "
          f"VALIDATION steps {steps[0]}-{steps[1]}: {len(y):,} transactions, {int(y.sum()):,} fraud ({y.mean():.4%})")
    flagged = scores >= approve
    print(f"\napprove_threshold {approve}: not auto-approved {int(flagged.sum()):,} | fraud caught "
          f"{int((flagged & y).sum()):,} | legit flagged {int((flagged & ~y).sum()):,} | "
          f"fraud approved {int((~flagged & y).sum()):,}")

    print(f"\ndecline candidates (approve_threshold fixed at {approve}):")
    print(f"{'decline_t':>9} {'declined':>9} {'legit_decl':>11} {'fraud_decl':>11} {'review_vol':>11} "
          f"{'fraud_in_rev':>13} {'precision':>10}")
    for r in decline_table(scores, y, approve):
        mark = "  <- configured" if r["decline_threshold"] == decline else ""
        print(f"{r['decline_threshold']:>9} {r['declined']:>9,} {r['legit_declined']:>11,} {r['fraud_declined']:>11,} "
              f"{r['review_volume']:>11,} {r['fraud_in_review']:>13,} {r['decline_precision']:>10.3f}{mark}")

    print("\ncalibration (observed fraud rate per score band):")
    for c in calibration(scores, y):
        print(f"  {c['band']:<14} n={c['n']:>9,}  mean score {c['mean_score']:.4f}  fraud rate {c['fraud_rate']:.4f}")


if __name__ == "__main__":
    main()
