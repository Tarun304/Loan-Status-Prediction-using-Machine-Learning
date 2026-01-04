"""FastAPI application for Loan Status Prediction"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .predict import predictor
from .schemas import HealthResponse, LoanInput, LoanPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("Starting Loan Status Prediction API")
    logger.info("=" * 70)

    if not predictor.load_artifacts():
        logger.error("Model artifacts failed to load. API will be unhealthy.")
    else:
        logger.info("API ready")
        logger.info(f"Model alias : {predictor.model_alias}")
        logger.info(f"Run ID      : {predictor.run_id}")

    yield

    logger.info("Shutting down API")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    status = predictor.get_artifacts_status()
    loaded = all(status.values())
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
    )


@app.post("/predict", response_model=LoanPrediction)
def predict(loan: LoanInput):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return LoanPrediction(**predictor.predict(loan.model_dump()))


@app.get("/")
def root():
    return {
        "message": "Loan Status Prediction API",
        "docs": "/docs",
        "health": "/health",
    }
