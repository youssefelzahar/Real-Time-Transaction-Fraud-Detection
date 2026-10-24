"""Load versioned artifacts and score transactions with the same FeaturePipeline used in training."""
import json
import os
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from time import perf_counter

from src.data_preprocessing import FeaturePipeline
from src.decision_engine import decide

# Per-request timing print; set TIMING_LOG=0 to turn it off (e.g. during load tests).
TIMING_LOG = os.getenv("TIMING_LOG", "1") == "1"


class FraudPredictor:
    def __init__(self, model, pipeline, metadata):
        booster_features = model.get_booster().feature_names
        if booster_features != pipeline.feature_names:
            raise ValueError(f"Model features {booster_features} != pipeline features {pipeline.feature_names}")
        self.model = model
        self.booster = model.get_booster()
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])
        self.feature_names = pipeline.feature_names
        self._model_version = f"{metadata['feature_set'].lower()}_{metadata['version']}"

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        model = XGBClassifier()
        model.load_model(model_dir / "model.json")
        # One row per request: multi-threading only adds overhead and oversubscribes
        # CPUs when several API workers run. Scale with workers instead.
        model.set_params(n_jobs=1)
        model.get_booster().set_param({"nthread": 1})
        pipeline = FeaturePipeline.load(model_dir / "pipeline.json")
        metadata = json.loads((model_dir / "metadata.json").read_text())
        return cls(model, pipeline, metadata)

    @property
    def model_version(self):
        return self._model_version

    def features(self, records):
        """Batch path (DataFrame), same as training."""
        return self.pipeline.transform(pd.DataFrame.from_records(records))

    def predict_proba(self, records):
        return self.model.predict_proba(self.features(records))[:, 1]

    def score(self, record):
        """Single-record hot path: no DataFrame, numpy straight into the booster.
        Column order is guaranteed by the schema check in __init__."""
        t0 = perf_counter()
        x = self.pipeline.transform_one(record)
        feature_ms = (perf_counter() - t0) * 1000

        t1 = perf_counter()
        probability = float(self.booster.inplace_predict(x.reshape(1, -1))[0])
        model_ms = (perf_counter() - t1) * 1000

        t2 = perf_counter()
        missing = [name for name, v in zip(self.feature_names, x) if v != v]
        decision = decide(probability, self.threshold)
        response = {
            "fraud_probability": probability,
            "decision": decision,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "missing_features": missing,
        }
        postprocess_ms = (perf_counter() - t2) * 1000

        if TIMING_LOG:
            print({
                "feature_ms": round(feature_ms, 3),
                "model_ms": round(model_ms, 3),
                "postprocess_ms": round(postprocess_ms, 3),
                "score_total_ms": round(feature_ms + model_ms + postprocess_ms, 3),
            })

        return response

