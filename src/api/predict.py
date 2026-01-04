"""Prediction logic for loan approval model.

This module handles model loading from MLflow registry and inference preprocessing
to match the training pipeline exactly.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LoanPredictor:
    """Handles loan approval prediction using MLflow-registered XGBoost model.

    Preprocessing pipeline matches training exactly:
    1. Box-Cox transformation on skewed features
    2. Feature engineering (total_assets creation)
    3. Categorical encoding
    4. Feature selection
    5. Standardization

    Note: Winsorization is applied during training on the full dataset but not
    during inference, as percentile-based clipping requires population statistics.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.boxcox_lambdas = None
        self.encodings = None
        self.run_id = None
        self.model_alias = None

    def load_artifacts(self) -> bool:
        """Load model and preprocessing artifacts from MLflow registry.

        Returns:
            bool: True if all artifacts loaded successfully, False otherwise.
        """
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            client = mlflow.MlflowClient()

            # Resolve champion alias to specific model version
            mv = client.get_model_version_by_alias(settings.MODEL_NAME, "champion")
            self.run_id = mv.run_id
            self.model_alias = "champion"

            logger.info(f"Resolved champion alias to run_id={self.run_id}")

            # Load model from registry
            model_uri = f"models:/{settings.MODEL_NAME}@champion"
            logger.info(f"Loading model from {model_uri}")
            self.model = mlflow.xgboost.load_model(model_uri)
            logger.info("Model loaded successfully")

            # Download preprocessing artifacts from MLflow run
            base_dir = Path(tempfile.mkdtemp())
            artifacts_dir = client.download_artifacts(self.run_id, "", str(base_dir))

            self._load_local_artifacts(Path(artifacts_dir))
            return True

        except Exception as e:
            logger.exception("Failed to load artifacts from MLflow")
            return False

    def _load_local_artifacts(self, root: Path):
        """Load preprocessing artifacts from downloaded MLflow artifacts directory.

        Args:
            root: Path to the root directory containing artifacts.
        """
        # Load StandardScaler
        self.scaler = joblib.load(root / "scaler.pkl")

        # Load selected features list
        with open(root / "selected_features.json") as f:
            self.selected_features = json.load(f)["features"]

        # Load Box-Cox lambda parameters
        with open(root / "boxcox_lambdas.json") as f:
            self.boxcox_lambdas = json.load(f)

        # Load categorical encoding mappings
        with open(root / "encodings.json") as f:
            self.encodings = json.load(f)

        logger.info("All preprocessing artifacts loaded successfully")

    def preprocess_input(self, features: Dict) -> pd.DataFrame:
        """Preprocess input features to match training pipeline.

        Applies the following transformations in order:
        1. Box-Cox transformation on skewed numerical features
        2. Create total_assets from sum of transformed asset values
        3. Drop individual asset columns
        4. Encode categorical features (ordinal and one-hot)
        5. Select final feature subset
        6. Apply standardization

        Args:
            features: Dictionary containing raw feature values.

        Returns:
            DataFrame with preprocessed and scaled features ready for prediction.
        """
        df = pd.DataFrame([features])

        # Step 1: Apply Box-Cox transformation to handle skewness
        # Use saved lambda parameters from training
        for col, lam in self.boxcox_lambdas.items():
            if col in df.columns:
                if lam == 0:
                    df[col] = np.log(df[col] + 1)
                else:
                    df[col] = ((df[col] + 1) ** lam - 1) / lam
                logger.debug(
                    f"Box-Cox transformation applied to {col} with lambda={lam}"
                )

        # Step 2: Create aggregated total_assets feature from transformed values
        assets = [
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ]
        df["total_assets"] = df[assets].sum(axis=1)
        logger.debug("Created total_assets feature")

        # Step 3: Drop individual asset columns as they are now aggregated
        df.drop(columns=assets, inplace=True, errors="ignore")

        # Step 4: Encode categorical features
        # Ordinal encoding for education level
        df["education"] = df["education"].map(self.encodings["education_mapping"])

        # One-hot encoding for employment_type and self_employed
        df = pd.get_dummies(
            df,
            columns=["employment_type", "self_employed"],
            drop_first=True,
        )

        # Ensure all expected dummy columns exist (set to 0 if missing)
        for col in [
            "employment_type_Freelancer",
            "employment_type_Salaried",
            "self_employed_Yes",
        ]:
            if col not in df.columns:
                df[col] = 0

        # Step 5: Select final feature subset used in training
        df = df[self.selected_features]
        logger.debug(f"Selected features: {self.selected_features}")

        # Step 6: Apply standardization using fitted scaler from training
        scaled_data = self.scaler.transform(df)
        df = pd.DataFrame(scaled_data, columns=self.selected_features)

        return df

    def predict(self, features: Dict) -> Dict:
        """Generate loan approval prediction.

        Args:
            features: Dictionary containing applicant information.

        Returns:
            Dictionary with prediction status, approval probability, and model version.
        """
        X = self.preprocess_input(features)
        proba = self.model.predict_proba(X)[0][1]
        pred = int(self.model.predict(X)[0])

        return {
            "status": "Approved" if pred == 1 else "Rejected",
            "probability": float(proba),
            "model_version": self.model_alias,
        }

    def get_artifacts_status(self) -> dict:
        """Check if all required artifacts are loaded.

        Returns:
            Dictionary indicating which artifacts are successfully loaded.
        """
        return {
            "model": self.model is not None,
            "scaler": self.scaler is not None,
            "features": self.selected_features is not None,
            "lambdas": self.boxcox_lambdas is not None,
            "encodings": self.encodings is not None,
        }


# Global predictor instance
predictor = LoanPredictor()
