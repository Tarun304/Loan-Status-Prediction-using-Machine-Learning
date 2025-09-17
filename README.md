### Loan Status Prediction – ML Model, FastAPI Backend, and Streamlit Frontend

Build, deploy, and use an end‑to‑end loan application decisioning system powered by an XGBoost classifier. The project exposes a REST API (FastAPI) for predictions and a user‑friendly Streamlit web app. Both are deployed and publicly accessible.

- Live API: [loan-prediction-api-1hq4.onrender.com](https://loan-prediction-api-1hq4.onrender.com)
- Live App: [loan-prediction-frontend.onrender.com](https://loan-prediction-frontend.onrender.com)
- Source Code: [Tarun304/Loan-Status-Prediction-using-Machine-Learning](https://github.com/Tarun304/Loan-Status-Prediction-using-Machine-Learning)

---

## Overview

This project predicts whether a loan should be Approved or Rejected, along with a confidence score. The pipeline:

1) Streamlit frontend collects applicant and financial inputs
2) FastAPI backend validates inputs and calls a prediction service
3) Prediction service loads preprocessing artifacts and an XGBoost model, transforms features, and returns a decision and probability

Key tech: FastAPI, Streamlit, Pandas, NumPy, scikit‑learn, XGBoost, joblib.

---

## Project Structure

```
.
├─ models/
│  ├─ lambdas.joblib              # Box‑Cox lambda values per numeric field
│  ├─ scaler.joblib               # Fitted scaler for model inputs
│  └─ xgboost_model.pkl           # Trained XGBoost classifier
├─ src/
│  ├─ api/
│  │  ├─ fastapi_app.py           # FastAPI app factory
│  │  ├─ models.py                # Pydantic request/response schemas
│  │  └─ routes.py                # API routes (/, /health, /predict)
│  ├─ backend/
│  │  └─ prediction_service.py    # Feature transforms and predict logic
│  └─ frontend/
│     └─ streamlit_app.py         # Streamlit UI that calls the API
├─ Jupyter Notebook_Loan Status Prediction_ Tarun Kumar Behera.ipynb
│                                # End-to-end EDA, preprocessing, training, and artifact export
├─ requirements.txt
└─ README.md
```

---

## Architecture

- FastAPI service (`src/api`)
  - Endpoints:
    - `GET /` – service banner and status
    - `GET /health` – health check
    - `POST /predict` – accepts a `LoanInput`, returns `LoanPrediction`
  - Uses `LoanPredictor` from `src/backend/prediction_service.py`

- Prediction service (`src/backend/prediction_service.py`)
  - Loads preprocessing artifacts from `models/`:
    - Box‑Cox lambdas (`lambdas.joblib`)
    - Feature scaler (`scaler.joblib`)
    - XGBoost model (`xgboost_model.pkl`)
  - Computes engineered features:
    - Box‑Cox transforms for amounts
    - `total_assets` = sum of asset values
    - One‑hot for `employment_type_Salaried`
  - Final model features (in order):
    - `cibil_score`, `loan_term`, `loan_amount`, `total_assets`, `income_annum`, `employment_type_Salaried`

- Streamlit app (`src/frontend/streamlit_app.py`)
  - Collects inputs with validation and calls `POST /predict`
  - Displays an Approved/Rejected decision and confidence
  - Points to the hosted API: `https://loan-prediction-api-1hq4.onrender.com`

---

## Model Development (Jupyter Notebook)

The notebook `Jupyter Notebook_Loan Status Prediction_ Tarun Kumar Behera.ipynb` is the foundation of the project. It performs:

1) Exploratory Data Analysis (EDA)
   - Inspect shapes, dtypes, head/sample; confirm no missing or duplicate rows
   - Visualizations: target distribution, categorical vs target, histograms, box plots, pair plot, correlation heatmap
   - Observation: target is imbalanced (Approved > Rejected); CIBIL strongly correlates with approval

2) Data Ingestion
   - CSV path used in notebook: `/content/loan_approval_data.csv` (dataset not committed in repo)

3) Data Cleaning and Transformations
   - Drop identifier: `loan_id`
   - Winsorization (by class) to cap outliers:
     - Columns: `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `bank_asset_value`
     - Limits: lower 5%, upper 10% (`limits=[0.05, 0.1]`, `inclusive=(True, True)`) applied separately for Approved/Rejected
   - Box‑Cox to reduce right‑skewness on monetary columns:
     - Columns: `loan_amount`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
     - Shift: add 1 before transform to avoid zeros; save fitted lambdas to `models/lambdas.joblib`
   - Feature engineering:
     - `total_assets` = `commercial_assets_value + residential_assets_value + bank_asset_value + luxury_assets_value`
     - Drop individual asset columns after aggregation
   - Categorical encoding:
     - Ordinal map for `education`: `{8th:0, 10th:1, 12th:2, Graduate:3}`
     - One‑hot with drop‑first for `employment_type`, `self_employed` → columns include `employment_type_Freelancer`, `employment_type_Salaried`, `self_employed_Yes` (cast to int)
   - Target encoding: `loan_status` → 1 (Approved), 0 (Rejected)
   - Shuffle with `random_state=42`

4) Train/Test Split and Class Imbalance Handling
   - Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)`
   - Oversampling: `SMOTE(random_state=42)` on training set

5) Feature Selection and Scaling
   - Random Forest feature importance on SMOTE data ranks features:
     - Top 6 retained: `cibil_score`, `loan_term`, `loan_amount`, `total_assets`, `income_annum`, `employment_type_Salaried`
   - Scaling: `StandardScaler()` fitted on training (SMOTE) subset of selected features
     - Save to `models/scaler.joblib`

6) Modeling and Hyperparameter Tuning
   - Candidate models: Logistic Regression, SVC, RandomForestClassifier, XGBClassifier
   - GridSearchCV (cv=5, scoring='f1') on SMOTE‑scaled training set:
     - RandomForest params: `n_estimators=[50,100,200]`, `max_depth=[None,10,20,30]`, `min_samples_split=[2,5,10]`
     - XGBoost params: `n_estimators=[50,100,200]`, `max_depth=[None,3,5,7]`, `learning_rate=[0.01,0.1,0.2]`, `subsample=[0.5,0.75,1.0]`, `colsample_bytree=[0.5,0.75,1.0]`
     - Best found: RF `{n_estimators:200, max_depth:None, min_samples_split:2}`; XGB `{n_estimators:200, max_depth:3, learning_rate:0.2, subsample:0.75, colsample_bytree:0.75}`

7) Evaluation
   - Metrics on test set (scaled with train scaler): Accuracy, Precision, Recall, F1
   - Learning curves (`sklearn.model_selection.learning_curve`) with scoring='accuracy' (cv=5)
   - Cross‑validation: `cross_val_score` (cv=5, scoring='f1') on SMOTE‑scaled training set
   - Selection: XGBoost chosen for best F1 (mean CV) and stability (lowest std)

8) Artifact Export
   - Save XGBoost model: `models/xgboost_model.pkl`
   - Save Box‑Cox lambdas: `models/lambdas.joblib`
   - Save scaler: `models/scaler.joblib`

These artifacts are consumed by the FastAPI prediction service at runtime.

---

## API

Base URL (live): `https://loan-prediction-api-1hq4.onrender.com`

### Request schema – `LoanInput`

```
{
  "no_of_dependents": int,
  "education": "8th" | "10th" | "12th" | "Graduate",
  "self_employed": "Yes" | "No",
  "employment_type": "Salaried" | "Business" | "Freelancer",
  "income_annum": number,
  "loan_amount": number,
  "loan_term": number,           // years
  "cibil_score": number,         // 300–900
  "residential_assets_value": number,
  "commercial_assets_value": number,
  "luxury_assets_value": number,
  "bank_asset_value": number
}
```

### Response schema – `LoanPrediction`

```
{
  "status": "Approved" | "Rejected",
  "probability": number // 0.0–1.0
}
```

### Example

```bash
curl -X POST \
  https://loan-prediction-api-1hq4.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
        "no_of_dependents": 1,
        "education": "Graduate",
        "self_employed": "No",
        "employment_type": "Salaried",
        "income_annum": 1200000,
        "loan_amount": 500000,
        "loan_term": 10,
        "cibil_score": 760,
        "residential_assets_value": 800000,
        "commercial_assets_value": 0,
        "luxury_assets_value": 0,
        "bank_asset_value": 150000
      }'
```

---

## Running Locally

Prerequisites: Python 3.10+ recommended.

1) Create and activate a virtual environment

```bash
python -m venv venv
# Windows PowerShell
./venv/Scripts/Activate.ps1
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Start the FastAPI backend (development)

```bash
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Basic checks:

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

4) Start the Streamlit frontend

By default the app points to the hosted API. To use your local API, update `API_URL` in `src/frontend/streamlit_app.py` to `http://localhost:8000`.

```bash
streamlit run src/frontend/streamlit_app.py
```

The app will open at the URL printed by Streamlit (commonly `http://localhost:8501`).

---

## Deployment

- Backend: Deployed (Render) at [loan-prediction-api-1hq4.onrender.com](https://loan-prediction-api-1hq4.onrender.com)
- Frontend: Deployed (Render) at [loan-prediction-frontend.onrender.com](https://loan-prediction-frontend.onrender.com)

Ensure the `models/` directory (with `lambdas.joblib`, `scaler.joblib`, and `xgboost_model.pkl`) is available to the server at runtime. On platforms like Render, add them to the repository or a persistent disk.

---

## Notes on Feature Engineering

- Box‑Cox transform applied to: `loan_amount`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
- `total_assets` is the sum of asset features after transform
- One‑hot `employment_type_Salaried` is used; other categories are encoded as 0

---

## Troubleshooting

- 422 Unprocessable Entity: Input schema mismatch – check field names and allowed string values
- 500 Server Error: Typically indicates missing model artifacts or shape mismatch – ensure `models/` exists and versions align with training
- Connection errors in Streamlit: Verify `API_URL` and that the backend is reachable

---

## License

This project is provided for educational purposes. Add a license if you intend to redistribute or use commercially.


