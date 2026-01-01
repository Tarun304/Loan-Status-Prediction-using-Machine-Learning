import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class DataPreprocessing:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize DataPreprocessing with parameters."""
        self.params = self.load_params(params_path)
        self.config = self.params["data_preprocessing"]
        self.encoders = {}

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

    def load_data(self) -> pd.DataFrame:
        """Load interim data from CSV."""
        try:
            input_path = self.config["interim_data_file"]
            df = pd.read_csv(input_path)
            logger.info("Data loaded from %s. Shape: %s", input_path, df.shape)
            return df
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        try:
            initial_shape = df.shape
            missing_count = df.isnull().sum().sum()

            if missing_count == 0:
                logger.info("No missing values found in the dataset")
                return df

            # Drop rows with missing target
            df = df.dropna(subset=["loan_status"])

            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.debug(
                        f"Filled {col} missing values with median: {median_val}"
                    )

            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=["object"]).columns
            categorical_cols = categorical_cols.drop("loan_status")

            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    logger.debug(f"Filled {col} missing values with mode: {mode_val}")

            logger.info(
                "Missing values handled. Shape: %s -> %s", initial_shape, df.shape
            )
            return df
        except Exception as e:
            logger.error("Failed to handle missing values: %s", e)
            raise

    def treat_outliers_by_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Winsorization separately for Approved and Rejected loans."""
        try:
            columns = self.config["winsorize_columns"]
            limits_approved = self.config["winsorize_limits_approved"]
            limits_rejected = self.config["winsorize_limits_rejected"]

            for column in columns:
                # Winsorization for Approved loans
                approved_mask = df["loan_status"] == "Approved"
                df.loc[approved_mask, column] = winsorize(
                    df.loc[approved_mask, column],
                    limits=limits_approved,
                    inclusive=(True, True),
                )

                # Winsorization for Rejected loans
                rejected_mask = df["loan_status"] == "Rejected"
                df.loc[rejected_mask, column] = winsorize(
                    df.loc[rejected_mask, column],
                    limits=limits_rejected,
                    inclusive=(True, True),
                )

                logger.debug(f"Winsorization applied to {column}")

            logger.info(
                "Outlier treatment completed using Winsorization on columns: %s",
                columns,
            )
            return df
        except Exception as e:
            logger.error("Failed to treat outliers: %s", e)
            raise

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        try:
            # Ordinal encoding for education
            education_order = {"8th": 0, "10th": 1, "12th": 2, "Graduate": 3}
            df["education"] = df["education"].replace(education_order)
            logger.debug(f"Education encoded: {education_order}")

            # One-hot encoding for employment_type and self_employed
            df = pd.get_dummies(
                df, columns=["employment_type", "self_employed"], drop_first=True
            )

            # Convert boolean to int
            if "employment_type_Freelancer" in df.columns:
                df["employment_type_Freelancer"] = df[
                    "employment_type_Freelancer"
                ].astype(int)
            if "employment_type_Salaried" in df.columns:
                df["employment_type_Salaried"] = df["employment_type_Salaried"].astype(
                    int
                )
            if "self_employed_Yes" in df.columns:
                df["self_employed_Yes"] = df["self_employed_Yes"].astype(int)

            logger.info("Categorical features encoded successfully")
            return df
        except Exception as e:
            logger.error("Failed to encode categorical features: %s", e)
            raise

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode target variable: Approved=1, Rejected=0."""
        try:
            target_mapping = {"Approved": 1, "Rejected": 0}
            df["loan_status"] = df["loan_status"].map(target_mapping)

            logger.info("Target variable encoded: %s", target_mapping)
            return df
        except Exception as e:
            logger.error("Failed to encode target: %s", e)
            raise

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets with stratification."""
        try:
            test_size = self.config["test_size"]
            random_state = self.config["random_state"]

            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df["loan_status"],
            )

            logger.info(
                "Data split completed (stratified). Train shape: %s, Test shape: %s",
                train_df.shape,
                test_df.shape,
            )
            return train_df, test_df
        except Exception as e:
            logger.error("Failed to split data: %s", e)
            raise

    def save_processed_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """Save processed train and test data."""
        try:
            output_path = self.config["processed_data_path"]
            os.makedirs(output_path, exist_ok=True)

            train_path = os.path.join(output_path, "train.csv")
            test_path = os.path.join(output_path, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info("Processed data saved to %s", output_path)
        except Exception as e:
            logger.error("Failed to save processed data: %s", e)
            raise

    def save_artifacts(self) -> None:
        """Save preprocessing artifacts."""
        try:
            artifacts_dir = "models/artifacts"
            os.makedirs(artifacts_dir, exist_ok=True)

            # Save encoding mappings
            encoding_info = {
                "education_mapping": {"8th": 0, "10th": 1, "12th": 2, "Graduate": 3},
                "target_mapping": {"Approved": 1, "Rejected": 0},
            }

            encodings_path = os.path.join(artifacts_dir, "encodings.json")
            with open(encodings_path, "w") as f:
                json.dump(encoding_info, f, indent=4)
            logger.info("Encoding mappings saved to %s", encodings_path)

        except Exception as e:
            logger.error("Failed to save artifacts: %s", e)
            raise

    def run(self):
        """Execute the data preprocessing pipeline."""
        try:
            logger.info("Starting data preprocessing pipeline...")

            # Load interim data
            df = self.load_data()

            # Handle missing values
            df = self.handle_missing_values(df)

            # Treat outliers (class-wise Winsorization)
            df = self.treat_outliers_by_class(df)

            # Encode categorical features
            df = self.encode_categorical_features(df)

            # Encode target variable
            df = self.encode_target(df)

            # Split into train/test (stratified)
            train_df, test_df = self.split_data(df)

            # Save processed data
            self.save_processed_data(train_df, test_df)

            # Save artifacts
            self.save_artifacts()

            logger.info("Data preprocessing completed successfully")

            # Display summary
            print("\n" + "=" * 60)
            print("PREPROCESSING SUMMARY")
            print("=" * 60)
            print(f"Train shape: {train_df.shape}")
            print(f"Test shape: {test_df.shape}")
            print(f"\nTarget distribution (train):")
            print(train_df["loan_status"].value_counts())
            print(f"\nTarget distribution (test):")
            print(test_df["loan_status"].value_counts())
            print("\nSample of processed train data:")
            print(train_df.head())

        except Exception as e:
            logger.error("Data preprocessing failed: %s", e)
            raise


if __name__ == "__main__":
    preprocessing = DataPreprocessing()
    preprocessing.run()
