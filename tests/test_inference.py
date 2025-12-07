from pathlib import Path

import pytest

from inference import InferenceModel
from train_model import TrainModel


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
    """
    Create a temporary trained model for inference tests.
    This uses TrainModel to train once per test session.
    """
    tmp_dir = tmp_path_factory.mktemp("models_test")
    model_dir = tmp_dir
    model_path = model_dir / "test_inference_model.joblib"

    trainer = TrainModel(
        data_path=Path("data/online_course_data.csv"),
        model_dir=model_dir,
        model_filename=model_path.name,
        test_size=0.2,
        random_state=42,
    )

    trainer.load_data()
    trainer.prepare_data_and_pipeline()
    trainer.train()
    trainer.save_model()

    return model_path


def test_inference_single_prediction(trained_model_path):
    """
    Sanity test for InferenceModel.predict_one().
    """
    model = InferenceModel(model_path=trained_model_path)

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

    assert "prediction" in result
    assert isinstance(result["prediction"], int)

    if "probability" in result:
        assert 0.0 <= result["probability"] <= 1.0
