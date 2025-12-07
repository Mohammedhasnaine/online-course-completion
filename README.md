ML Model Deployment: Course Completion Prediction (FastAPI, Docker, AWS)
Overview

This project implements an end-to-end machine learning system for predicting whether a student will complete an online course.
It includes a full workflow covering:

Dataset preprocessing

Model training using a Random Forest classifier

Model serialization

Model storage on AWS S3

Inference pipeline using a custom inference class

FastAPI application for serving predictions

Dockerization for containerized deployment

Deployment on AWS ECS (Fargate) with ECR

Unit tests using Pytest

The project is structured for real-world MLOps workflows and demonstrates how to deploy ML models as scalable API services.

Features

Training pipeline implemented as a class (TrainModel)

Inference pipeline implemented as a class (InferenceModel)

FastAPI-based REST API with endpoints:

/health

/predict

/predict_batch

Fully containerized with Docker

Production deployment on AWS ECS via ECR container images

Model artifact stored on AWS S3

Automated preprocessing built into the inference pipeline

Unit testing using Pytest

Clear directory structure for maintainability

Project Architecture
flowchart TD

A[Dataset: CSV File] --> B[TrainModel Class]
B --> C[Preprocessing + Feature Engineering]
C --> D[Random Forest Training]
D --> E[Model Artifact (.joblib)]
E --> F[S3 Model Storage]

E --> G[InferenceModel Class]
G --> H[FastAPI Application]

H --> I[Docker Container]
I --> J[ECR Repository]
J --> K[ECS Fargate Task]
K --> L[Public API via Load Balancer]

Repository Structure
online-course-completion/
 ├── app/
 │    ├── __init__.py
 │    └── main.py
 ├── data/
 │    └── online_course_data.csv
 ├── docs/
 │    └── images/         (place your ECS screenshots here)
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

Setup Instructions
1. Install Dependencies (Using Poetry)
poetry install

2. Activate Poetry Environment
poetry shell

Training the Model

The training pipeline reads the dataset, applies preprocessing, trains a Random Forest model, evaluates it, and saves the model artifact.

To run:

poetry run python train_model.py

S3 Model Upload

The training script also supports uploading the trained model to S3 using:

trainer.upload_model_to_s3(
    model_path=model_path,
    bucket_name="your-bucket-name",
    object_name="random_forest_pipeline.joblib"
)


Your S3 bucket structure:

s3://course-completion-models-<your initials>/random_forest_pipeline.joblib

Inference Pipeline

InferenceModel automatically:

Loads the saved model pipeline

Performs preprocessing identical to training

Produces predictions + probabilities

Example usage:

model = InferenceModel()
result = model.predict_one(sample_features)

Running the FastAPI Application

Start the API locally:

poetry run uvicorn app.main:app --reload


Now open:

Swagger UI: http://localhost:8000/docs

Health check: http://localhost:8000/health

API Endpoints
POST /predict

Input example:

{
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

Docker Usage
Build Image
docker build -t course-completion-api .

Run with Docker Compose
docker-compose up

Access API

http://localhost:8000/docs

AWS Deployment Summary (ECR + ECS Fargate)
1. Push Image to ECR
aws ecr get-login-password --region ap-south-1 \
| docker login --username AWS --password-stdin <your ECR URI>

docker build -t course-completion-api .
docker tag course-completion-api:latest <ECR_URI>:latest
docker push <ECR_URI>:latest

2. ECS Fargate Deployment Steps

Create ECS Cluster

Create Task Definition (Fargate)

Add container using ECR image

Set container port to 8000

Create ECS Service

Create Application Load Balancer

Add Target Group and health check at /health

Deploy and access API via ALB DNS

Deployment Screenshot Placeholders

Upload your ECS screenshots to:

docs/images/


Then reference them like this:

![ECS Cluster](docs/images/ecs-cluster.png)
![ECS Service](docs/images/ecs-service.png)
![ALB](docs/images/alb.png)


Replace filenames with your actual uploaded names.

Running Unit Tests
pytest


Covers:

Training pipeline tests

Inference pipeline tests

FastAPI endpoint tests

Future Improvements

Load model dynamically from S3 during inference

Add CI/CD (GitHub Actions)

Add model monitoring and drift detection

Implement automatic retraining pipeline

Add feature importance dashboards

Uploading ECS Screenshots to GitHub

Go to your GitHub repo

Navigate to:
docs/images/

Click:

Add file → Upload files

Select your ECS screenshots

Commit changes

Then the images automatically appear in README if filenames match.

##  Author

**Name:** Mohammed Hasnaine  
**Course:** B.E. Final Year  
**Project Type:** ML Classification Project  
**GitHub:** [@Mohammedhasnaine](https://github.com/Mohammedhasnaine)
