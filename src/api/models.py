from pydantic import BaseModel
from typing import Literal

# Define the request schema
class LoanInput(BaseModel):
    no_of_dependents: int
    education: Literal["8th", "10th", "12th", "Graduate"]
    self_employed: Literal["Yes", "No"]
    employment_type: Literal["Salaried", "Business", "Freelancer"]
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

# Define the response schema
class LoanPrediction(BaseModel):
    status: Literal["Approved", "Rejected"]
    probability: float
