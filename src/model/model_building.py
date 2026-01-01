import logging
import os
import pickle

import pandas as pd
import yaml
from xgboost import XGBClassifier

# Logging configuration
logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class ModelBuilder:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize ModelBuilder with parameters."""
        self.params = self.load_params(params_path)
        self.config = self.params["model_building"]
        self.model = None

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

    def load_processed_data(self) -> tuple:
        """Load processed training and validation data."""
        try:
            train_df = pd.read_csv(self.config["train_features_file"])
            test_df = pd.read_csv(self.config["test_features_file"])

            # Separate features and target
            X_train = train_df.drop("loan_status", axis=1)
            y_train = train_df["loan_status"]
            X_test = test_df.drop("loan_status", axis=1)
            y_test = test_df["loan_status"]

            logger.info(
                "Processed data loaded. X_train: %s, y_train: %s",
                X_train.shape,
                y_train.shape,
            )
            logger.info("X_test: %s, y_test: %s", X_test.shape, y_test.shape)

            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error("Failed to load processed data: %s", e)
            raise

    def build_model(self) -> XGBClassifier:
        """Build XGBoost model with configured hyperparameters."""
        try:
            xgb_params = self.config["xgboost_params"]
            random_state = self.config["random_state"]

            model = XGBClassifier(random_state=random_state, **xgb_params)

            logger.info("XGBoost model initialized with parameters:")
            for key, value in xgb_params.items():
                logger.info("  %s: %s", key, value)

            return model
        except Exception as e:
            logger.error("Failed to build model: %s", e)
            raise

    def train_model(self, X_train, y_train) -> None:
        """Train the XGBoost model."""
        try:
            logger.info("Starting model training...")

            self.model.fit(X_train, y_train)

            logger.info("Model training completed")

        except Exception as e:
            logger.error("Failed to train model: %s", e)
            raise

    def save_model(self) -> None:
        """Save the trained model."""
        try:
            model_output_path = self.config["model_output_path"]
            model_name = self.config["model_name"]

            os.makedirs(model_output_path, exist_ok=True)

            model_path = os.path.join(model_output_path, model_name)

            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            logger.info("Model saved to %s", model_path)

        except Exception as e:
            logger.error("Failed to save model: %s", e)
            raise

    def run(self):
        """Execute the model building pipeline."""
        try:
            logger.info("Starting model building...")

            # Load data
            X_train, y_train, X_test, y_test = self.load_processed_data()

            # Build model
            self.model = self.build_model()

            # Train model
            self.train_model(X_train, y_train)

            # Save model
            self.save_model()

            logger.info("Model building completed successfully")

            # Display summary
            print("\n" + "=" * 60)
            print("MODEL BUILDING SUMMARY")
            print("=" * 60)
            print(f"Model Type: XGBoost Classifier")
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Features: {X_train.shape[1]}")
            print(
                f"Model saved to: {self.config['model_output_path']}/{self.config['model_name']}"
            )

        except Exception as e:
            logger.error("Model building failed: %s", e)
            raise


if __name__ == "__main__":
    builder = ModelBuilder()
    builder.run()
