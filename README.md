# ML Model Deployment: Course Completion Prediction (FastAPI, Docker, AWS)

## Overview

This project implements an end-to-end machine learning system that predicts whether a student will complete an online course.  
It demonstrates a complete ML Engineering and MLOps workflow, including:

- Data preprocessing  
- Model training using a Random Forest classifier  
- Model artifact storage in AWS S3  
- Inference pipeline with consistent preprocessing  
- REST API deployment using FastAPI  
- Docker containerization  
- Cloud deployment using AWS ECR and ECS Fargate  
- Automated testing using Pytest  

This repository reflects a production-style architecture where the model, training pipeline, inference logic, and deployment stack are fully separated and modular.

## Features

- Class-based training pipeline (`TrainModel`) with preprocessing, model training, evaluation, and artifact saving  
- Inference pipeline (`InferenceModel`) that loads the trained model and performs consistent preprocessing  
- FastAPI application exposing prediction endpoints with automatic Swagger documentation  
- Dockerfile and Docker Compose support for containerized API deployment  
- AWS ECR integration for container image storage  
- AWS ECS Fargate deployment workflow for scalable cloud hosting  
- S3 integration for storing trained model artifacts  
- Comprehensive unit tests using Pytest (training, inference, API)  
- Structured, maintainable folder layout suitable for real-world ML projects

## Architecture Diagram

```markdown
## Architecture Diagram

flowchart TD

    A[Dataset (CSV)] --> B[TrainModel Class]
    B --> C[Preprocessing + Feature Engineering]
    C --> D[Train Random Forest Model]
    D --> E[Model Artifact (.joblib)]
    E --> F[Upload to S3]

    E --> G[InferenceModel Class]
    G --> H[FastAPI Application]

    H --> I[Docker Container]
    I --> J[ECR Repository]
    J --> K[ECS Fargate Task]
    K --> L[Public API via Load Balancer]

## Directory Structure

    online-course-completion/
     ├── app/
     │    ├── __init__.py
     │    └── main.py
     ├── data/
     │    └── online_course_data.csv
     ├── docs/
     │    └── images/        # Upload ECS deployment screenshots
     ├── models/
     │    └── random_forest_pipeline.joblib
     ├── tests/
     │    ├── conftest.py
     │    ├── test_api.py
     │    ├── test_inference.py
     │    └── test_training.py
     ├── Dockerfile
     ├── docker-compose.yml
     ├── inference.py
     ├── train_model.py
     ├── requirements.txt
     ├── requirements-api.txt
     ├── pyproject.toml
     ├── poetry.lock
     └── README.md

## Environment Setup

### Install Dependencies (Poetry)

Install all project dependencies:

poetry install


### Activate the Virtual Environment

Activate the Poetry environment so all Python commands use correct dependencies:

poetry shell

You are now ready to run training, inference, tests, or the FastAPI application inside 

## Training the Model

The project uses a class-based training pipeline implemented in `TrainModel` inside `train_model.py`.  
Training performs:

- Reading the dataset  
- Splitting into train/test sets  
- Preprocessing numeric and categorical features  
- Training a Random Forest classifier  
- Evaluating the model  
- Saving the trained model locally  
- Uploading the model artifact to AWS S3 (optional)

### Run the Training Script

poetry run python train_model.py


After running, the trained model is saved at:



models/random_forest_pipeline.joblib


### Uploading Model to S3

The training script contains a helper method to upload the trained model to your S3 bucket:

```python
trainer.upload_model_to_s3(
    model_path=model_path,
    bucket_name="course-completion-models-<your initials>",
    object_name="random_forest_pipeline.joblib"
)


Model artifacts stored in S3 allow versioning, remote access, and integration into cloud inference workflows.

## Inference Pipeline

Inference is handled by the `InferenceModel` class in `inference.py`.  
This class ensures the same preprocessing used during training is consistently applied during prediction.

### Key Responsibilities

- Load the trained model pipeline  
- Validate and preprocess incoming feature data  
- Predict class label (0 or 1)  
- Return prediction probability  

### Example Usage

```python
from inference import InferenceModel

model = InferenceModel()

sample = {
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
    "videos_watched_pct": 80
}

result = model.predict_one(sample)
print(result)

Output example:

{
  "prediction": 1,
  "probability": 0.82
}

## Running the FastAPI Application

The API is implemented in `app/main.py` and wraps the `InferenceModel` to provide prediction endpoints.

### Start the API Locally

Run the following command:
poetry run uvicorn app.main:app --reload

This inference pipeline is also integrated directly into the FastAPI application for serving real-time predictions.


### Available Endpoints

#### GET /health
Used to verify the service is running.

#### GET /docs
Opens the automatically generated Swagger UI.

#### POST /predict
Accepts a single student's feature values and returns a prediction.

#### POST /predict_batch
Accepts multiple records for batch prediction.

### Sample /predict Request

```json
{
  "age": 22,
  "hours_per_week": 6,
  "num_logins_last_month": 15,
  "assignments_submitted": 4,
  "discussion_posts": 3,
  "num_siblings": 2,
  "continent": "Asia",
  "education_level": "Bachelors",
  "preferred_device": "Laptop",
  "has_pet": 0,
  "is_working_professional": 1,
  "videos_watched_pct": 70
}

Running the FastAPI server locally allows easy testing before packaging the application into a Docker container.

## Docker Usage

The project includes a Dockerfile and a docker-compose configuration to containerize the FastAPI inference service.

### Build the Docker Image

Run the following:

docker build -t course-completion-api .


This creates a production-ready image containing:

- FastAPI application  
- Inference pipeline  
- Python environment and dependencies  

### Run the Service Using Docker Compose

docker-compose up

This starts the API inside a container and exposes it on port 8000.

### Access the API

After the container is running, open:

http://localhost:8000/docs

The Swagger documentation will allow testing prediction endpoints interactively.

Containerization ensures the application runs consistently across any machine or cloud environme

## AWS Deployment (ECR + ECS Fargate)

This project was deployed to AWS using a production-grade architecture involving:

- Amazon ECR for storing the Docker image  
- Amazon ECS (Fargate) for serverless container hosting  
- Application Load Balancer for public API access  
- IAM roles for secure permissions  
- CloudWatch for logs and monitoring  

### Step 1: Push Docker Image to ECR

Authenticate Docker to ECR:

aws ecr get-login-password --region ap-south-1
| docker login --username AWS --password-stdin <ECR_URI>

Build and tag the image:

docker build -t course-completion-api .
docker tag course-completion-api:latest <ECR_URI>:latest

Push to ECR:

docker push <ECR_URI>:latest


### Step 2: Deploy Using ECS Fargate

The deployment included:

- Creating an ECS Cluster  
- Registering a Task Definition  
- Adding the container from ECR  
- Exposing port 8000  
- Creating a Service that maintains the container  
- Attaching an Application Load Balancer  
- Setting health check endpoint to `/health`  
- Testing the publicly accessible API via the ALB DNS name  

### Deployment Screenshots

Upload your deployment images to:

docs/images/

Then reference them in the README:

Once images are uploaded, they will be displayed automatically in this section.

## Running Unit Tests

This project includes Pytest-based unit tests to ensure reliability of:

- The training pipeline  
- The inference pipeline  
- The FastAPI application  

All tests are located in the `tests/` directory.

### Run All Tests

pytest

### Test Coverage

1. **test_training.py**  
   - Verifies that the training pipeline runs successfully  
   - Ensures the model file is created  

2. **test_inference.py**  
   - Checks that the inference model loads correctly  
   - Validates prediction output format  

3. **test_api.py**  
   - Tests `/health` endpoint  
   - Tests `/predict` and `/predict_batch` endpoints  
   - Ensures API responds with correct structure and status codes  

Tests help ensure consistency across updates and deployments.

## Future Improvements

Several enhancements can be added to evolve this project into a full production-grade MLOps system:

### 1. Load Model Directly from S3 During Inference
Instead of loading the model from the local `models/` directory, the inference service can:

- Download the latest model from S3 at startup  
- Cache it locally  
- Allow automatic model updates without redeployment  

This enables rapid iteration and real-time model versioning.

### 2. Automated Model Retraining
A retraining pipeline can be scheduled using:

- AWS Lambda  
- AWS Step Functions  
- CloudWatch EventBridge  

Triggered by data drift or periodic intervals.

### 3. Model Drift and Data Drift Monitoring
Production systems typically monitor:

- Statistical drift in input features  
- Decrease in prediction accuracy  
- Change in data distributions  

Tools like Evidently AI or AWS SageMaker Model Monitor can be integrated.

### 4. CI/CD Pipeline
Add GitHub Actions for:

- Automated testing  
- Docker image builds  
- Deployment triggers  

Ensures reproducibility and stable updates.

### 5. Feature Importance and Explainability
Generate:

- SHAP values  
- Feature importances  
- Model interpretability dashboards  

Useful for debugging and presenting insights.

### 6. Frontend Dashboard
Optional UI to:

- Submit prediction inputs  
- Display completion probability  
- Visualize student risk levels  

### 7. Logging and Monitoring Enhancements
Extend:

- Structured logging  
- API performance metrics  
- Error tracking  

to improve observability in production.

These improvements can evolve the project into a full-scale production-ready ML platform.

## How to Upload Deployment Screenshots

To document your AWS deployment (ECR, ECS, Load Balancer, Tasks), you can upload screenshots and reference them inside this README.

### Step 1: Prepare Your Images

Name your screenshots clearly, for example:

ecs-cluster.png
ecs-service.png
ecs-task.png
alb.png


### Step 2: Upload to GitHub

1. Open your GitHub repository in the browser  
2. Navigate to the folder:

docs/images/


3. Click:

Add file → Upload files


4. Select your screenshots  
5. Click **Commit changes**

### Step 3: Display Screenshots in README

Use the following Markdown syntax:

```markdown
![ECS Cluster](docs/images/ecs-cluster.png)
![ECS Service](docs/images/ecs-service.png)
![ECS Task](docs/images/ecs-task.png)
![Application Load Balancer](docs/images/alb.png)

Once uploaded, your screenshots will automatically appear in the README in the AWS deployment section.

##  Author

**Name:** Mohammed Hasnaine  
**Course:** B.E. Final Year  
**Project Type:** ML Classification Project  
**GitHub:** [@Mohammedhasnaine](https://github.com/Mohammedhasnaine)
