from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import ConfigDict

from inference import InferenceModel


# ---------- Pydantic Schemas ----------

class StudentFeatures(BaseModel):
    """
    Request body schema for a single student's features.

    NOTE:
    - Fields are optional so you can experiment.
    - Extra fields are allowed and will be passed to the model.
    """
    age: Optional[int] = None
    hours_per_week: Optional[float] = None
    num_logins_last_month: Optional[int] = None
    assignments_submitted: Optional[int] = None
    discussion_posts: Optional[int] = None
    num_siblings: Optional[int] = None

    continent: Optional[str] = None
    education_level: Optional[str] = None
    preferred_device: Optional[str] = None

    has_pet: Optional[int] = None
    is_working_professional: Optional[int] = None
    videos_watched_pct: Optional[float] = None

    # This makes FastAPI accept extra fields without rejecting the request.
    model_config = ConfigDict(extra="allow")


class PredictionResponse(BaseModel):
    """
    Response schema for a single prediction.
    """
    prediction: int
    probability: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch predictions.
    """
    results: List[PredictionResponse]


# ---------- FastAPI App ----------

app = FastAPI(
    title="Online Course Completion Prediction API",
    description="Predict whether a student will complete the course.",
    version="1.0.0",
)

# Initialize the inference model once when the app starts.
# This loads models/random_forest_pipeline.joblib
inference_model = InferenceModel()


# ---------- Routes / Endpoints ----------

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: StudentFeatures) -> PredictionResponse:
    """
    Predict completion for a single student.

    You send JSON with student features,
    the API returns prediction + probability.
    """
    # Convert pydantic model to plain dict
    features_dict = features.model_dump()
    result = inference_model.predict_one(features_dict)
    # FastAPI will automatically convert dict -> PredictionResponse
    return PredictionResponse(**result)


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    batch: List[StudentFeatures],
) -> BatchPredictionResponse:
    """
    Predict completion for multiple students at once.

    Input: list of feature objects.
    Output: list of prediction results.
    """
    batch_dicts = [item.model_dump() for item in batch]
    results = inference_model.predict_batch(batch_dicts)
    pred_objects = [PredictionResponse(**r) for r in results]
    return BatchPredictionResponse(results=pred_objects)
