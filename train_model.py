import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class TrainModel:
    def __init__(self, model_name, data_path="data/online_course_completion.csv", target_col="completed_course"):
        self.model_name = model_name
        self.data_path = data_path
        self.target_col = target_col
        self.data = None
        self.model = None

    def load_data(self):
        print(f"📂 Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)

        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        print("✅ Data loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")

        # Print NaN summary
        nan_counts = self.data.isna().sum()
        if nan_counts.sum() > 0:
            print("\n⚠️ Missing values detected:")
            print(nan_counts[nan_counts > 0])
        else:
            print("\n✅ No missing values detected!")

    def _build_model(self):
        # Split numeric vs categorical features
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()

        if self.target_col in numeric_features:
            numeric_features.remove(self.target_col)
        if self.target_col in categorical_features:
            categorical_features.remove(self.target_col)

        # Pipelines for numeric + categorical
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Choose model
        if self.model_name == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
        elif self.model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_name == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Final pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', model)])
        return clf

    def train(self):
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("⚡ Training model...")
        clf = self._build_model()
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        print(f"✅ Model trained successfully! Accuracy: {score:.4f}")

        self.model = clf

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{self.model_name}.pkl"
        joblib.dump(self.model, model_path)
        print(f"💾 Model saved at {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for course completion prediction")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to train: logistic_regression | random_forest | xgboost")
    args = parser.parse_args()

    trainer = TrainModel(model_name=args.model)
    trainer.load_data()
    trainer.train()
    trainer.save_model()

