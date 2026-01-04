"""Utility functions for Streamlit app - Calls FastAPI for predictions"""

import os
from typing import Dict

import requests

# Get API URL from environment variable (for Docker) or use localhost (for local dev)
API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict_via_api(features: Dict) -> Dict:
    """
    Call FastAPI to make loan prediction

    Args:
        features: Dictionary with loan application features

    Returns:
        Dictionary with status, probability, and model_version
    """
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result

    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Cannot connect to API at {API_URL}. Make sure FastAPI is running!"
        )
    except requests.exceptions.Timeout:
        raise Exception("API request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"API error: {e.response.text}")
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def check_api_health() -> bool:
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# Feature information for UI
FEATURE_INFO = {
    "no_of_dependents": {
        "name": "Number of Dependents",
        "desc": "Total number of family members dependent on applicant",
        "range": (0, 10),
        "default": 2,
        "type": "int",
    },
    "education": {
        "name": "Education Level",
        "desc": "Highest education qualification",
        "options": ["8th", "10th", "12th", "Graduate"],
        "default": "Graduate",
        "type": "select",
    },
    "self_employed": {
        "name": "Self Employment Status",
        "desc": "Whether applicant is self-employed",
        "options": ["Yes", "No"],
        "default": "No",
        "type": "select",
    },
    "employment_type": {
        "name": "Employment Type",
        "desc": "Type of employment",
        "options": ["Salaried", "Business", "Freelancer"],
        "default": "Salaried",
        "type": "select",
    },
    "income_annum": {
        "name": "Annual Income (₹)",
        "desc": "Total annual income in Indian Rupees",
        "range": (100000.0, 100000000.0),
        "default": 600000.0,
        "type": "float",
    },
    "loan_amount": {
        "name": "Loan Amount (₹)",
        "desc": "Requested loan amount in Indian Rupees",
        "range": (100000.0, 100000000.0),
        "default": 5000000.0,
        "type": "float",
    },
    "loan_term": {
        "name": "Loan Term (Years)",
        "desc": "Duration of loan repayment in years",
        "range": (1.0, 30.0),
        "default": 12.0,
        "type": "float",
    },
    "cibil_score": {
        "name": "CIBIL Score",
        "desc": "Credit score ranging from 300 to 900",
        "range": (300.0, 900.0),
        "default": 750.0,
        "type": "float",
    },
    "residential_assets_value": {
        "name": "Residential Assets Value (₹)",
        "desc": "Total value of residential properties owned",
        "range": (0.0, 100000000.0),
        "default": 8000000.0,
        "type": "float",
    },
    "commercial_assets_value": {
        "name": "Commercial Assets Value (₹)",
        "desc": "Total value of commercial properties owned",
        "range": (0.0, 100000000.0),
        "default": 2000000.0,
        "type": "float",
    },
    "luxury_assets_value": {
        "name": "Luxury Assets Value (₹)",
        "desc": "Total value of luxury items (cars, jewelry, etc.)",
        "range": (0.0, 50000000.0),
        "default": 1500000.0,
        "type": "float",
    },
    "bank_asset_value": {
        "name": "Bank Assets Value (₹)",
        "desc": "Total value of bank deposits and liquid assets",
        "range": (0.0, 100000000.0),
        "default": 3000000.0,
        "type": "float",
    },
}
