"""Test data pipeline components"""

import os

import pandas as pd
import pytest
import yaml


def test_raw_data_exists():
    """Test that raw data file exists"""
    assert os.path.exists(
        "data/raw/loan_approval_data.csv"
    ), "Raw data not found at data/raw/loan_approval_data.csv"


def test_params_yaml_exists():
    """Test that params.yaml exists"""
    assert os.path.exists("params.yaml"), "params.yaml not found"


def test_params_yaml_structure():
    """Test params.yaml has required sections"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    assert "data_ingestion" in params, "data_ingestion section missing"
    assert "data_preprocessing" in params, "data_preprocessing section missing"
    assert "feature_engineering" in params, "feature_engineering section missing"
    assert "model_building" in params, "model_building section missing"
    assert "model_evaluation" in params, "model_evaluation section missing"

    # Test data_ingestion parameters
    assert "raw_data_file" in params["data_ingestion"]
    assert "interim_data_path" in params["data_ingestion"]
    assert "drop_columns" in params["data_ingestion"]

    # Test data_preprocessing parameters
    assert "test_size" in params["data_preprocessing"]
    assert "random_state" in params["data_preprocessing"]
    assert "winsorize_columns" in params["data_preprocessing"]
    assert "boxcox_columns" in params["data_preprocessing"]
    assert "asset_columns" in params["data_preprocessing"]

    # Test feature_engineering parameters
    assert "selected_features" in params["feature_engineering"]
    assert "apply_smote" in params["feature_engineering"]

    # Test model_building parameters
    assert "xgboost_params" in params["model_building"]


def test_interim_data_after_ingestion():
    """Test interim data is created after data ingestion"""
    if os.path.exists("data/interim/loan_data_cleaned.csv"):
        df = pd.read_csv("data/interim/loan_data_cleaned.csv")

        assert df.shape[0] > 0, "Interim data is empty"
        assert "loan_status" in df.columns, "Target column missing"
        assert "loan_id" not in df.columns, "loan_id should be dropped"


def test_processed_data_after_preprocessing():
    """Test processed data is created after preprocessing"""
    if os.path.exists("data/processed/train.csv") and os.path.exists(
        "data/processed/test.csv"
    ):
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")

        assert train_df.shape[0] > 0, "Training data is empty"
        assert test_df.shape[0] > 0, "Test data is empty"
        assert "loan_status" in train_df.columns, "Target column missing in train"
        assert "loan_status" in test_df.columns, "Target column missing in test"

        # Check that individual asset columns are dropped
        asset_cols = [
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ]
        for col in asset_cols:
            assert col not in train_df.columns, f"{col} should be dropped"
            assert col not in test_df.columns, f"{col} should be dropped"

        # Check that total_assets is created
        assert "total_assets" in train_df.columns, "total_assets should be created"


def test_feature_engineered_data():
    """Test feature engineered data is created after feature engineering"""
    if os.path.exists("data/features/train_features.csv") and os.path.exists(
        "data/features/test_features.csv"
    ):
        X_train = pd.read_csv("data/features/train_features.csv")
        X_test = pd.read_csv("data/features/test_features.csv")

        assert X_train.shape[0] > 0, "X_train is empty"
        assert X_test.shape[0] > 0, "X_test is empty"
        assert "loan_status" in X_train.columns, "Target should be in features file"

        # Check that only selected features are present
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        selected_features = params["feature_engineering"]["selected_features"]

        for feature in selected_features:
            assert feature in X_train.columns, f"{feature} missing from train features"
            assert feature in X_test.columns, f"{feature} missing from test features"


def test_artifacts_exist():
    """Test that preprocessing artifacts exist"""
    if os.path.exists("models/artifacts"):
        artifacts = [
            "scaler.pkl",
            "selected_features.json",
            "boxcox_lambdas.json",
            "encodings.json",
        ]

        for artifact in artifacts:
            artifact_path = os.path.join("models/artifacts", artifact)
            if os.path.exists(artifact_path):
                assert (
                    os.path.getsize(artifact_path) > 0
                ), f"{artifact} exists but is empty"


def test_train_test_split_ratio():
    """Test that train-test split ratio matches params"""
    if os.path.exists("data/processed/train.csv") and os.path.exists(
        "data/processed/test.csv"
    ):
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        test_size = params["data_preprocessing"]["test_size"]

        total_size = len(train_df) + len(test_df)
        actual_test_ratio = len(test_df) / total_size

        # Allow 5% tolerance
        assert (
            abs(actual_test_ratio - test_size) < 0.05
        ), f"Test split ratio {actual_test_ratio} doesn't match expected {test_size}"


def test_class_balance_in_split():
    """Test that train-test split maintains class balance"""
    if os.path.exists("data/processed/train.csv") and os.path.exists(
        "data/processed/test.csv"
    ):
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")

        train_balance = train_df["loan_status"].mean()
        test_balance = test_df["loan_status"].mean()

        # Allow 10% difference in class balance
        assert (
            abs(train_balance - test_balance) < 0.1
        ), "Class imbalance between train and test sets"
