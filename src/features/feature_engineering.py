import json
import logging
import os
import pickle

import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class FeatureEngineering:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize FeatureEngineering with parameters."""
        self.params = self.load_params(params_path)
        self.config = self.params["feature_engineering"]
        self.scaler = None

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

    def load_data(self) -> tuple:
        """Load preprocessed train and test data."""
        try:
            train_path = self.config["train_file"]
            test_path = self.config["test_file"]

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Train data loaded. Shape: %s", train_df.shape)
            logger.info("Test data loaded. Shape: %s", test_df.shape)

            return train_df, test_df
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            raise

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        """Apply SMOTE to balance classes in training data."""
        try:
            if not self.config.get("apply_smote", True):
                logger.info("SMOTE is disabled in config")
                return X_train, y_train

            random_state = self.config["random_state"]

            # Check class distribution before SMOTE
            before_counts = y_train.value_counts().to_dict()
            logger.info("Class distribution before SMOTE: %s", before_counts)

            # Apply SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            # Check class distribution after SMOTE
            after_counts = pd.Series(y_train_smote).value_counts().to_dict()
            logger.info("Class distribution after SMOTE: %s", after_counts)
            logger.info("Training data shape after SMOTE: %s", X_train_smote.shape)

            return X_train_smote, y_train_smote
        except Exception as e:
            logger.error("Failed to apply SMOTE: %s", e)
            raise

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Scale selected features using StandardScaler."""
        try:
            # Get selected features from params
            selected_features = self.config["selected_features"]
            logger.info("Using selected features from params: %s", selected_features)

            # Validate features exist
            missing_train = set(selected_features) - set(X_train.columns)
            missing_test = set(selected_features) - set(X_test.columns)

            if missing_train:
                raise ValueError(f"Features missing in train data: {missing_train}")
            if missing_test:
                raise ValueError(f"Features missing in test data: {missing_test}")

            # Select only those features
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            # Fit scaler on train data
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_selected)
            X_test_scaled = self.scaler.transform(X_test_selected)

            # Convert back to DataFrame
            X_train_scaled = pd.DataFrame(
                X_train_scaled, columns=selected_features, index=X_train_selected.index
            )
            X_test_scaled = pd.DataFrame(
                X_test_scaled, columns=selected_features, index=X_test_selected.index
            )

            logger.info("Features scaled using StandardScaler")
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.error("Failed to scale features: %s", e)
            raise

    def save_engineered_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Save feature engineered data."""
        try:
            output_path = self.config["output_path"]
            os.makedirs(output_path, exist_ok=True)

            # Combine features and target
            train_final = X_train.copy()
            train_final["loan_status"] = y_train.values

            test_final = X_test.copy()
            test_final["loan_status"] = y_test.values

            # Save
            train_path = os.path.join(output_path, "train_features.csv")
            test_path = os.path.join(output_path, "test_features.csv")

            train_final.to_csv(train_path, index=False)
            test_final.to_csv(test_path, index=False)

            logger.info("Feature engineered data saved to %s", output_path)
        except Exception as e:
            logger.error("Failed to save engineered data: %s", e)
            raise

    def save_artifacts(self) -> None:
        """Save feature engineering artifacts for inference."""
        try:
            artifacts_dir = "models/artifacts"
            os.makedirs(artifacts_dir, exist_ok=True)

            # Save scaler
            scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info("StandardScaler saved to %s", scaler_path)

            # Save selected features list
            selected_features = self.config["selected_features"]
            features_path = os.path.join(artifacts_dir, "selected_features.json")
            with open(features_path, "w") as f:
                json.dump({"features": selected_features}, f, indent=4)
            logger.info("Selected features saved to %s", features_path)

        except Exception as e:
            logger.error("Failed to save artifacts: %s", e)
            raise

    def run(self):
        """Execute the feature engineering pipeline."""
        try:
            logger.info("Starting feature engineering pipeline...")

            # Load preprocessed data (already has Box-Cox, total_assets, etc.)
            train_df, test_df = self.load_data()

            # Separate features and target
            X_train = train_df.drop("loan_status", axis=1)
            y_train = train_df["loan_status"]
            X_test = test_df.drop("loan_status", axis=1)
            y_test = test_df["loan_status"]

            # Apply SMOTE on training data
            X_train_smote, y_train_smote = self.apply_smote(X_train, y_train)

            # Scale selected features (use SMOTE data for training)
            X_train_scaled, X_test_scaled = self.scale_features(
                pd.DataFrame(X_train_smote, columns=X_train.columns), X_test
            )

            # Save engineered data
            self.save_engineered_data(
                X_train_scaled, X_test_scaled, y_train_smote, y_test
            )

            # Save artifacts
            self.save_artifacts()

            logger.info("Feature engineering completed successfully")

            # Display summary
            print("\n" + "=" * 60)
            print("FEATURE ENGINEERING SUMMARY")
            print("=" * 60)
            print(f"Train shape (after SMOTE + scaling): {X_train_scaled.shape}")
            print(f"Test shape (scaled): {X_test_scaled.shape}")
            print(f"\nSelected features: {self.config['selected_features']}")
            print(f"\nTarget distribution (train after SMOTE):")
            print(pd.Series(y_train_smote).value_counts())
            print(f"\nTarget distribution (test):")
            print(y_test.value_counts())
            print("\nSample of engineered train data:")
            print(X_train_scaled.head())

        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            raise


if __name__ == "__main__":
    feature_eng = FeatureEngineering()
    feature_eng.run()
