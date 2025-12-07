from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class InferenceModel:
    """
    Class for loading the trained model pipeline and performing inference.

    This class:
    - Loads the joblib-saved pipeline (preprocessing + model)
    - Detects which feature columns the pipeline expects
    - Accepts raw feature dictionaries (same fields as training)
    - Returns predictions and optional probabilities
    """

    model_path: Path = Path("models/random_forest_pipeline.joblib")

    # internal fields, not passed from outside
    pipeline: Optional[Pipeline] = None
    expected_feature_cols: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Load model pipeline and extract expected columns."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        print(f"[INFO] Loading model pipeline from: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

        if not isinstance(self.pipeline, Pipeline):
            raise TypeError(
                f"Loaded object is not a sklearn Pipeline. Got: {type(self.pipeline)}"
            )

        # Extract expected columns from the ColumnTransformer inside the pipeline
        preprocessor = self.pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            raise ValueError("Pipeline does not contain a 'preprocessor' step.")

        transformers = preprocessor.transformers_
        # We assumed in training: ("num", numeric_transformer, numeric_cols)
        #                         ("cat", categorical_transformer, categorical_cols)
        numeric_cols = []
        categorical_cols = []

        for name, transformer, cols in transformers:
            if name == "num":
                numeric_cols = list(cols)
            elif name == "cat":
                categorical_cols = list(cols)

        self.expected_feature_cols = list(dict.fromkeys(numeric_cols + categorical_cols))

        print("[INFO] Model pipeline loaded successfully.")
        print(f"[INFO] Expected feature columns: {self.expected_feature_cols}")

    def _to_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert a single feature dictionary to a one-row DataFrame and
        align it with the expected columns of the pipeline.
        """
        if not isinstance(features, dict):
            raise ValueError("Features must be provided as a dictionary.")

        # Start from user-provided features
        df = pd.DataFrame([features])

        # Ensure all expected columns exist; if missing, add as NaN
        for col in self.expected_feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Drop any extra columns that the pipeline doesn't know about
        extra_cols = [c for c in df.columns if c not in self.expected_feature_cols]
        if extra_cols:
            print(f"[WARN] Dropping unexpected columns: {extra_cols}")
            df = df.drop(columns=extra_cols)

        # Reorder columns to match training order
        df = df[self.expected_feature_cols]

        return df

    def predict_one(
        self, features: Dict[str, Any]
    ) -> Dict[str, Union[int, float, bool]]:
        """
        Predict for a single sample.

        Returns a dictionary with:
        - prediction: int
        - probability: float (probability of class '1', if supported)
        """

        if self.pipeline is None:
            raise ValueError("Pipeline is not loaded.")

        # Convert input features to DataFrame aligned with pipeline
        X = self._to_dataframe(features)

        # Predict class
        print(f"[INFO] Running prediction for a single sample: {features}")
        pred = self.pipeline.predict(X)[0]

        result: Dict[str, Union[int, float, bool]] = {"prediction": int(pred)}

        # If pipeline supports predict_proba, return probability of class '1'
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X)[0]
            # Assuming binary classification [prob_class_0, prob_class_1]
            if len(proba) == 2:
                result["probability"] = float(proba[1])
            else:
                # multi-class case: you can adapt this if needed
                result["probabilities"] = proba.tolist()

        print(f"[INFO] Prediction result: {result}")
        return result

    def predict_batch(
        self, batch_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Union[int, float, bool]]]:
        """
        Predict for a batch of samples.

        batch_features: List of dicts, each same format as predict_one input.
        Returns list of result dicts.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not loaded.")

        if not isinstance(batch_features, list):
            raise ValueError("batch_features must be a list of dictionaries.")

        print(f"[INFO] Running batch prediction for {len(batch_features)} samples.")

        # Build a DataFrame from list of dicts
        df = pd.DataFrame(batch_features)

        # Align with expected columns (like in _to_dataframe)
        for col in self.expected_feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        extra_cols = [c for c in df.columns if c not in self.expected_feature_cols]
        if extra_cols:
            print(f"[WARN] Dropping unexpected columns in batch: {extra_cols}")
            df = df.drop(columns=extra_cols)

        df = df[self.expected_feature_cols]

        preds = self.pipeline.predict(df)

        results: List[Dict[str, Union[int, float, bool]]] = []
        has_proba = hasattr(self.pipeline, "predict_proba")

        if has_proba:
            probas = self.pipeline.predict_proba(df)

        for idx, pred in enumerate(preds):
            res: Dict[str, Union[int, float, bool]] = {"prediction": int(pred)}
            if has_proba:
                proba_row = probas[idx]
                if len(proba_row) == 2:
                    res["probability"] = float(proba_row[1])
                else:
                    res["probabilities"] = proba_row.tolist()
            results.append(res)

        print("[INFO] Batch prediction completed.")
        return results


def _demo_single_prediction():
    """
    Simple demo function for manual testing from CLI.

    You can run:
        poetry run python inference.py
    to test this without FastAPI.
    """
    model = InferenceModel()

    # IMPORTANT: these keys should roughly match your dataset's feature columns
    sample_features = {
        "age": 25,
        "hours_per_week": 5,
        "num_logins_last_month": 10,
        "assignments_submitted": 3,
        "discussion_posts": 2,
        "num_siblings": 1,
        "continent": "Asia",
        "education_level": "Bachelors",
        "preferred_device": "Mobile",
        "has_pet": 1,
        "is_working_professional": 0,
        "videos_watched_pct": 80.0,
    }

    result = model.predict_one(sample_features)
    print("Demo prediction output:", result)


if __name__ == "__main__":
    _demo_single_prediction()
