import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainModel:
    """
    Class to handle training of the course completion model.

    Responsibilities:
    - Load data
    - Apply preprocessing (drop columns, impute, encode, scale)
    - Train a model (RandomForest)
    - Evaluate on a validation split
    - Save trained pipeline to the models/ directory
    """

    data_path: Path = Path("data/online_course_data.csv")
    target_col: str = "completed_course"
    model_dir: Path = Path("models")
    model_filename: str = "random_forest_pipeline.joblib"

    test_size: float = 0.2
    random_state: int = 42

    # Internal fields (not passed from CLI)
    df: Optional[pd.DataFrame] = field(default=None, init=False)
    pipeline: Optional[Pipeline] = field(default=None, init=False)
    X_train: Optional[pd.DataFrame] = field(default=None, init=False)
    X_test: Optional[pd.DataFrame] = field(default=None, init=False)
    y_train: Optional[pd.Series] = field(default=None, init=False)
    y_test: Optional[pd.Series] = field(default=None, init=False)

    def load_data(self) -> None:
        """Load CSV data into a pandas DataFrame."""
        print(f"[INFO] Loading data from: {self.data_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        print(f"[INFO] Data loaded with shape: {self.df.shape}")

    def _prepare_features_and_target(self):
        """Apply column drops and split into X (features) and y (target)."""
        if self.df is None:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")

        df = self.df.copy()

        # Columns to drop (from your README)
        drop_cols: List[str] = [
            "favorite_color",
            "birth_month",
            "height_cm",
            "weight_kg",
            "country",
        ]

        existing_drop_cols = [c for c in drop_cols if c in df.columns]
        if existing_drop_cols:
            print(f"[INFO] Dropping columns: {existing_drop_cols}")
            df = df.drop(columns=existing_drop_cols)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data.")

        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Build the preprocessing + model pipeline.

        Categorical (One-Hot Encode):
            - continent
            - education_level
            - preferred_device

        Numerical (StandardScaler):
            - age
            - hours_per_week
            - num_logins_last_month
            - assignments_submitted
            - discussion_posts
            - num_siblings
        """
        # Only keep columns that actually exist in X (defensive coding)
        categorical_cols = [
            c for c in ["continent", "education_level", "preferred_device"]
            if c in X.columns
        ]

        numeric_cols = [
            c
            for c in [
                "age",
                "hours_per_week",
                "num_logins_last_month",
                "assignments_submitted",
                "discussion_posts",
                "num_siblings",
            ]
            if c in X.columns
        ]

        print(f"[INFO] Categorical columns: {categorical_cols}")
        print(f"[INFO] Numeric columns: {numeric_cols}")

        # You may have other numeric/boolean columns (like is_working_professional, has_pet)
        # Optionally add them as numeric if needed:
        extra_numeric = [
            col
            for col in X.select_dtypes(include=["int64", "float64", "bool"]).columns
            if col not in numeric_cols and col not in categorical_cols
        ]
        if extra_numeric:
            print(f"[INFO] Treating additional numeric columns as numeric: {extra_numeric}")
            numeric_cols = numeric_cols + extra_numeric

        # Preprocessing for numerical data
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",  # drop any columns we didn't list
        )

        # Model (using class_weight='balanced' to handle imbalance)
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        print("[INFO] Pipeline successfully built.")
        return pipeline

    def prepare_data_and_pipeline(self) -> None:
        """Combine data prep, split, and pipeline building."""
        X, y = self._prepare_features_and_target()

        # Train/validation split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        print(
            f"[INFO] Train shape: {self.X_train.shape}, "
            f"Test shape: {self.X_test.shape}"
        )

        # Build the pipeline using the training features
        self.pipeline = self._build_pipeline(self.X_train)

    def train(self) -> None:
        """Fit the pipeline on the training data."""
        if self.pipeline is None:
            raise ValueError("Pipeline is not built. Call prepare_data_and_pipeline() first.")

        print("[INFO] Training the model...")
        self.pipeline.fit(self.X_train, self.y_train)
        print("[INFO] Training completed.")

    def evaluate(self) -> None:
        """Evaluate the model on the test set and print metrics."""
        if self.pipeline is None:
            raise ValueError("Pipeline is not trained. Call train() first.")
        print("[INFO] Evaluating the model on the test set...")
        y_pred = self.pipeline.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        print(f"[METRIC] Accuracy: {acc:.4f}")
        print("[METRIC] Classification report:")
        print(classification_report(self.y_test, y_pred))

    def save_model(self) -> Path:
        """Save the trained pipeline to the models/ directory."""
        if self.pipeline is None:
            raise ValueError("Pipeline is not trained. Call train() first.")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / self.model_filename

        joblib.dump(self.pipeline, model_path)
        print(f"[INFO] Saved trained model to: {model_path.resolve()}")
        return model_path

    def upload_model_to_s3(self, model_path: Path, bucket_name: str, object_name: str):
        import boto3

        s3_client = boto3.client("s3")
        s3_client.upload_file(str(model_path), bucket_name, object_name)

        print(f"[INFO] Uploaded model to S3: s3://{bucket_name}/{object_name}")


def parse_args():
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train the Online Course Completion model and save it to models/."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/online_course_data.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        default="random_forest_pipeline.joblib",
        help="Filename for the saved model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use as test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = TrainModel(
        data_path=Path(args.data_path),
        model_dir=Path(args.model_dir),
        model_filename=args.model_filename,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Step-by-step operations
    trainer.load_data()
    trainer.prepare_data_and_pipeline()
    trainer.train()
    trainer.evaluate()
    model_path = trainer.save_model()     

    trainer.upload_model_to_s3(           
        model_path=model_path,
        bucket_name="course-completion-models-mh",   # your bucket
        object_name="random_forest_pipeline.joblib"
    )


if __name__ == "__main__":
    main()

