"""Test FastAPI application components"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def test_import_app():
    """Test that FastAPI app can be imported"""
    try:
        from src.api.app import app

        assert app is not None
    except Exception as e:
        assert False, f"Failed to import app: {e}"


def test_pydantic_schemas():
    """Test Pydantic schema validation"""
    from src.api.schemas import HealthResponse, LoanInput, LoanPrediction

    # Test LoanInput with example data
    test_data = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "employment_type": "Salaried",
        "income_annum": 600000.0,
        "loan_amount": 5000000.0,
        "loan_term": 12.0,
        "cibil_score": 750.0,
        "residential_assets_value": 8000000.0,
        "commercial_assets_value": 2000000.0,
        "luxury_assets_value": 1500000.0,
        "bank_asset_value": 3000000.0,
    }

    features = LoanInput(**test_data)
    assert features.no_of_dependents == 2
    assert features.education == "Graduate"
    assert features.cibil_score == 750.0
    assert features.loan_amount == 5000000.0

    # Test LoanPrediction
    response = LoanPrediction(
        status="Approved", probability=0.95, model_version="champion"
    )
    assert response.status == "Approved"
    assert response.probability == 0.95
    assert response.model_version == "champion"

    # Test HealthResponse
    health = HealthResponse(status="healthy", model_loaded=True)
    assert health.status == "healthy"
    assert health.model_loaded is True


def test_api_config():
    """Test API configuration"""
    from src.api.config import settings

    assert settings.APP_NAME is not None
    assert settings.MODEL_NAME == "loan-approval-xgboost"
    assert settings.MLFLOW_TRACKING_URI is not None
    assert settings.VERSION == "1.0.0"
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000


def test_predictor_class():
    """Test LoanPredictor class can be instantiated"""
    from src.api.predict import LoanPredictor

    predictor = LoanPredictor()
    assert predictor.model is None  # Not loaded in tests
    assert hasattr(predictor, "load_artifacts")
    assert hasattr(predictor, "predict")
    assert hasattr(predictor, "preprocess_input")
    assert hasattr(predictor, "get_artifacts_status")


def test_pydantic_validation_errors():
    """Test that Pydantic validates input correctly"""
    from pydantic import ValidationError

    from src.api.schemas import LoanInput

    # Test invalid education value
    try:
        invalid_data = {
            "no_of_dependents": 2,
            "education": "InvalidEducation",  # Invalid value
            "self_employed": "No",
            "employment_type": "Salaried",
            "income_annum": 600000.0,
            "loan_amount": 5000000.0,
            "loan_term": 12.0,
            "cibil_score": 750.0,
            "residential_assets_value": 8000000.0,
            "commercial_assets_value": 2000000.0,
            "luxury_assets_value": 1500000.0,
            "bank_asset_value": 3000000.0,
        }
        LoanInput(**invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected

    # Test negative loan amount
    try:
        invalid_data = {
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "employment_type": "Salaried",
            "income_annum": 600000.0,
            "loan_amount": -1000,  # Invalid negative
            "loan_term": 12.0,
            "cibil_score": 750.0,
            "residential_assets_value": 8000000.0,
            "commercial_assets_value": 2000000.0,
            "luxury_assets_value": 1500000.0,
            "bank_asset_value": 3000000.0,
        }
        LoanInput(**invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected
