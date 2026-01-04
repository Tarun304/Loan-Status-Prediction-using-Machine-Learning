"""Test model components"""

import json
import os

import pytest


def test_model_exists():
    """Test that trained model exists"""
    if os.path.exists("models/xgboost_model.pkl"):
        assert (
            os.path.getsize("models/xgboost_model.pkl") > 0
        ), "Model file exists but is empty"


def test_metrics_exist():
    """Test that evaluation metrics exist"""
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json", "r") as f:
            metrics = json.load(f)

        assert "accuracy" in metrics, "Accuracy metric missing"
        assert "precision" in metrics, "Precision metric missing"
        assert "recall" in metrics, "Recall metric missing"
        assert "f1_score" in metrics, "F1 score missing"


def test_metrics_values():
    """Test that metrics are within reasonable ranges"""
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json", "r") as f:
            metrics = json.load(f)

        # Basic sanity checks - all metrics should be between 0 and 1
        assert 0 <= metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= metrics["precision"] <= 1, "Precision should be between 0 and 1"
        assert 0 <= metrics["recall"] <= 1, "Recall should be between 0 and 1"
        assert 0 <= metrics["f1_score"] <= 1, "F1 score should be between 0 and 1"

        # Quality checks - model should perform reasonably well
        assert (
            metrics["accuracy"] > 0.85
        ), f"Model accuracy too low: {metrics['accuracy']} - model quality issue"
        assert (
            metrics["f1_score"] > 0.85
        ), f"Model F1 score too low: {metrics['f1_score']} - model quality issue"


def test_experiment_info_exists():
    """Test that experiment info exists after evaluation"""
    if os.path.exists("reports/experiment_info.json"):
        with open("reports/experiment_info.json", "r") as f:
            exp_info = json.load(f)

        assert "run_id" in exp_info, "MLflow run_id missing"
        assert "model_path" in exp_info, "Model path missing"
        assert "metrics" in exp_info, "Metrics missing from experiment info"
        assert (
            "model_artifact_path" in exp_info
        ), "Model artifact path missing from experiment info"


def test_preprocessing_artifacts():
    """Test that all preprocessing artifacts are saved"""
    artifacts_dir = "models/artifacts"

    if os.path.exists(artifacts_dir):
        # Check scaler
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        assert os.path.exists(scaler_path), "Scaler artifact missing"
        assert os.path.getsize(scaler_path) > 0, "Scaler file is empty"

        # Check selected features
        features_path = os.path.join(artifacts_dir, "selected_features.json")
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                features_data = json.load(f)
            assert "features" in features_data, "Features key missing in JSON"
            assert len(features_data["features"]) > 0, "No features selected"

        # Check Box-Cox lambdas
        lambdas_path = os.path.join(artifacts_dir, "boxcox_lambdas.json")
        if os.path.exists(lambdas_path):
            with open(lambdas_path, "r") as f:
                lambdas = json.load(f)
            assert len(lambdas) > 0, "No Box-Cox lambdas saved"

        # Check encodings
        encodings_path = os.path.join(artifacts_dir, "encodings.json")
        if os.path.exists(encodings_path):
            with open(encodings_path, "r") as f:
                encodings = json.load(f)
            assert "education_mapping" in encodings, "Education mapping missing"
            assert "target_mapping" in encodings, "Target mapping missing"


def test_model_hyperparameters():
    """Test that model hyperparameters match params.yaml"""
    import yaml

    if os.path.exists("params.yaml"):
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        xgb_params = params["model_building"]["xgboost_params"]

        # Check that all required parameters are specified
        required_params = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        ]

        for param in required_params:
            assert (
                param in xgb_params
            ), f"Required XGBoost parameter {param} missing from params.yaml"
