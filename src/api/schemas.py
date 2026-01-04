"""Pydantic models for request/response validation"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoanInput(BaseModel):
    """Input features for loan prediction"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    no_of_dependents: int = Field(..., description="Number of dependents", ge=0)
    education: Literal["8th", "10th", "12th", "Graduate"] = Field(
        ..., description="Education level"
    )
    self_employed: Literal["Yes", "No"] = Field(
        ..., description="Self-employment status"
    )
    employment_type: Literal["Salaried", "Business", "Freelancer"] = Field(
        ..., description="Type of employment"
    )
    income_annum: float = Field(..., description="Annual income", gt=0)
    loan_amount: float = Field(..., description="Requested loan amount", gt=0)
    loan_term: float = Field(..., description="Loan term in months", gt=0)
    cibil_score: float = Field(..., description="CIBIL credit score", ge=300, le=900)
    residential_assets_value: float = Field(
        ..., description="Value of residential assets", ge=0
    )
    commercial_assets_value: float = Field(
        ..., description="Value of commercial assets", ge=0
    )
    luxury_assets_value: float = Field(..., description="Value of luxury assets", ge=0)
    bank_asset_value: float = Field(..., description="Bank asset value", ge=0)


class LoanPrediction(BaseModel):
    """Response model for loan prediction"""

    status: Literal["Approved", "Rejected"] = Field(
        ..., description="Loan approval status"
    )
    probability: float = Field(
        ..., description="Probability of approval (0-1)", ge=0, le=1
    )
    model_version: str = Field(
        ..., description="Model alias used (champion/challenger)"
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="API status: healthy or unhealthy")
    model_loaded: bool = Field(
        ..., description="Whether model is ready to serve predictions"
    )
