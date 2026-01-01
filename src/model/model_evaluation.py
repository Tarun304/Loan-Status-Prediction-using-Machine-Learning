import json
import logging
import os
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MLFLOW_TRACKING_URI

# Logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class ModelEvaluator:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize ModelEvaluator with parameters."""
        self.params = self.load_params(params_path)
        self.config = self.params["model_evaluation"]
        self.build_config = self.params["model_building"]
        self.model = None
        self.metrics = {}

    def load_params(self, params_path: str) -> dict:
        """Load parameters from YAML file."""
        try:
            with open(params_path, "r") as file:
                params = yaml.safe_load(file)
            logger.debug("Parameters retrieved from %s", params_path)
            return params
        except Exception as e:
            logger.error("Failed to load params: %s", e)
            raise

    def load_model(self) -> None:
        """Load trained XGBoost model."""
        try:
            model_path = self.config["model_path"]
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Model loaded from %s", model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def load_test_data(self) -> tuple:
        """Load test data."""
        try:
            test_df = pd.read_csv(self.config["test_features_file"])

            X_test = test_df.drop("loan_status", axis=1)
            y_test = test_df["loan_status"]

            logger.info(
                "Test data loaded. X_test: %s, y_test: %s", X_test.shape, y_test.shape
            )
            return X_test, y_test
        except Exception as e:
            logger.error("Failed to load test data: %s", e)
            raise

    def make_predictions(self, X_test) -> object:
        """Make predictions on test data."""
        try:
            y_pred = self.model.predict(X_test)
            logger.info("Predictions generated for %d samples", len(y_pred))
            return y_pred
        except Exception as e:
            logger.error("Failed to make predictions: %s", e)
            raise

    def calculate_metrics(self, y_test, y_pred) -> dict:
        """Calculate evaluation metrics."""
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }

            logger.info("Metrics calculated:")
            logger.info("  Accuracy:  %.6f", accuracy)
            logger.info("  Precision: %.6f", precision)
            logger.info("  Recall:    %.6f", recall)
            logger.info("  F1 Score:  %.6f", f1)

            # Additional detailed metrics
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)

            logger.info("\nConfusion Matrix:\n%s", cm)
            logger.info("\nClassification Report:\n%s", cr)

            return metrics
        except Exception as e:
            logger.error("Failed to calculate metrics: %s", e)
            raise

    def save_metrics(self, metrics: dict) -> None:
        """Save metrics to JSON file."""
        try:
            metrics_file = self.config["metrics_file"]
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info("Metrics saved to %s", metrics_file)
        except Exception as e:
            logger.error("Failed to save metrics: %s", e)
            raise

    def run(self):
        """Execute the model evaluation pipeline."""
        try:
            logger.info("Starting model evaluation...")

            with mlflow.start_run(run_name="model_evaluation"):
                # Log model building parameters
                xgb_params = self.build_config["xgboost_params"]
                for key, value in xgb_params.items():
                    mlflow.log_param(key, value)
                mlflow.log_param("random_state", self.build_config["random_state"])

                # Load model and data
                self.load_model()
                X_test, y_test = self.load_test_data()
                y_pred = self.make_predictions(X_test)
                self.metrics = self.calculate_metrics(y_test, y_pred)

                # Log XGBoost model
                logger.info("Logging XGBoost model to MLflow...")
                mlflow.xgboost.log_model(xgb_model=self.model, name="model")
                logger.info("XGBoost model logged successfully")

                # Log preprocessing artifacts
                logger.info("Logging preprocessing artifacts to MLflow...")
                mlflow.log_artifact("models/artifacts/scaler.pkl")
                mlflow.log_artifact("models/artifacts/boxcox_lambdas.json")
                mlflow.log_artifact("models/artifacts/selected_features.json")
                mlflow.log_artifact("models/artifacts/encodings.json")
                logger.info("All preprocessing artifacts logged to MLflow")

                # Log evaluation metrics
                mlflow.log_metric("test_accuracy", self.metrics["accuracy"])
                mlflow.log_metric("test_precision", self.metrics["precision"])
                mlflow.log_metric("test_recall", self.metrics["recall"])
                mlflow.log_metric("test_f1_score", self.metrics["f1_score"])

                # Save and log metrics file
                self.save_metrics(self.metrics)
                mlflow.log_artifact(self.config["metrics_file"])

                # Save experiment info
                run_id = mlflow.active_run().info.run_id
                experiment_info = {
                    "run_id": run_id,
                    "model_path": "model",
                    "metrics": self.metrics,
                }

                info_file = "reports/experiment_info.json"
                os.makedirs("reports", exist_ok=True)
                with open(info_file, "w") as f:
                    json.dump(experiment_info, f, indent=4)

                mlflow.log_artifact(info_file)

                logger.info("Model evaluation completed successfully")
                logger.info("MLflow run ID: %s", run_id)

                # Display summary
                print("\n" + "=" * 60)
                print("MODEL EVALUATION SUMMARY")
                print("=" * 60)
                print(f"Test samples: {len(y_test)}")
                print(f"\nTest Set Performance:")
                print(f"  Accuracy:  {self.metrics['accuracy']:.6f}")
                print(f"  Precision: {self.metrics['precision']:.6f}")
                print(f"  Recall:    {self.metrics['recall']:.6f}")
                print(f"  F1 Score:  {self.metrics['f1_score']:.6f}")
                print(f"\nMLflow Run ID: {run_id}")

                return self.metrics

        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
