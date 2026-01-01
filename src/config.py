import os

import mlflow
from dotenv import load_dotenv

# Load environment variables from .env file (for local dev)
load_dotenv()

# Get DagsHub credentials
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Get MLflow credentials
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Configure MLflow
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("loan-approval-prediction-pipeline")
    print(f"MLflow configured: {MLFLOW_TRACKING_URI}")
else:
    print("Warning: MLFLOW_TRACKING_URI not found")
