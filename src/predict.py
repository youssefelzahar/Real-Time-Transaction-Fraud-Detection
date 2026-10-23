"""Load versioned artifacts and score transactions with the same FeaturePipeline used in training."""
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.data_preprocessing import FeaturePipeline
from src.decision_engine import decide


class FraudPredictor:
    def __init__(self, model, pipeline, metadata):
        booster_features = model.get_booster().feature_names
        if booster_features != pipeline.feature_names:
            raise ValueError(f"Model features {booster_features} != pipeline features {pipeline.feature_names}")
        self.model = model
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        model = XGBClassifier()
        model.load_model(model_dir / "model.json")
        # One row per request: multi-threading only adds overhead and oversubscribes
        # CPUs when several API workers run. Scale with workers instead.
        model.set_params(n_jobs=1)
        pipeline = FeaturePipeline.load(model_dir / "pipeline.json")
        metadata = json.loads((model_dir / "metadata.json").read_text())
        return cls(model, pipeline, metadata)

    @property
    def model_version(self):
        return f"{self.metadata['feature_set'].lower()}_{self.metadata['version']}"

    def features(self, records):
        return self.pipeline.transform(pd.DataFrame.from_records(records))

    def predict_proba(self, records):
        return self.model.predict_proba(self.features(records))[:, 1]

    def score(self, record):
        X = self.features([record])
        probability = float(self.model.predict_proba(X)[:, 1][0])
        missing = [c for c in X.columns if pd.isna(X.iloc[0][c])]
        return {"fraud_probability": probability,
                "decision": decide(probability, self.threshold),
                "threshold": self.threshold,
                "model_version": self.model_version,
                "missing_features": missing}
