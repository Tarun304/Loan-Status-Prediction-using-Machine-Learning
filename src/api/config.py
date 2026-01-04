"""API Configuration Settings"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """API Configuration Settings"""

    # API Information
    APP_NAME: str = "Loan Status Prediction API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "MLOps-powered API for predicting loan approval status using XGBoost"
    )

    # MLflow/DagsHub Configuration
    MODEL_NAME: str = "loan-approval-xgboost"
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/tkbehera304/Loan-Status-Prediction-using-Machine-Learning.mlflow",
    )

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
