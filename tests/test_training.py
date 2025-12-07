from pathlib import Path

from train_model import TrainModel


def test_training_pipeline(tmp_path):
    """
    Basic sanity test:
    - Train the model using TrainModel
    - Save it into a temporary models/ directory
    - Check that the file exists
    """
    data_path = Path("data/online_course_data.csv")
    assert data_path.exists(), "Training CSV not found. Did you place it in data/?"

    # Use a temporary models dir so tests don't overwrite your real model
    model_dir = tmp_path / "models"

    trainer = TrainModel(
        data_path=data_path,
        model_dir=model_dir,
        model_filename="test_model.joblib",
        test_size=0.2,
        random_state=42,
    )

    trainer.load_data()
    trainer.prepare_data_and_pipeline()
    trainer.train()
    trainer.evaluate()
    model_path = trainer.save_model()

    assert model_path.exists(), "Model file was not saved."
