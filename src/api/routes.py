from fastapi import APIRouter, HTTPException
from src.backend.prediction_service import LoanPredictor
from src.api.models import LoanInput, LoanPrediction

router = APIRouter()
predictor = LoanPredictor()

@router.get("/")
def root():
    return {"message": "Loan Status Prediction API", "status": "running"}

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.post("/predict", response_model=LoanPrediction)
def predict_loan_status(loan_data: LoanInput):
    try:
        prediction = predictor.predict(loan_data.dict())
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
