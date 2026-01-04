# Loan Status Prediction - End-to-End MLOps Pipeline

A demonstration project showcasing a complete machine learning system for predicting loan approval status using XGBoost classifier. This project demonstrates MLOps best practices with automated model training, versioning, deployment, and monitoring.

## 🚀 Live Deployment

### Deployed Services

- **API Service**: [https://loan-api-latest.onrender.com/](https://loan-api-latest.onrender.com/)
  - API Documentation: [https://loan-api-latest.onrender.com/docs](https://loan-api-latest.onrender.com/docs)
  - Health Check: [https://loan-api-latest.onrender.com/health](https://loan-api-latest.onrender.com/health)

- **Streamlit Web Application**: [https://loan-streamlit-latest.onrender.com/](https://loan-streamlit-latest.onrender.com/)

### ⚠️ Important Notes on Deployment

**Free Tier Limitations:**
- Both services are deployed on Render's free tier instances
- Services will automatically shut down after 15 minutes of inactivity
- First request after inactivity may take 30-60 seconds to wake up the service
- The Streamlit app requires the API service to be active to function properly
- If the API is sleeping, the Streamlit app will show a connection error until the API wakes up

**Note:** This is a demonstration project. For actual production use, consider upgrading to paid tiers or using alternative hosting solutions for better reliability.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [MLOps Pipeline](#mlops-pipeline)
- [Model Development](#model-development)
- [API Documentation](#api-documentation)
- [Local Setup](#local-setup)
- [Running the Application](#running-the-application)
- [Deployment](#deployment)
- [Technology Stack](#technology-stack)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a complete end-to-end machine learning system for loan approval prediction. It combines:

- **Data Pipeline**: Automated data ingestion, preprocessing, and feature engineering using DVC
- **Model Training**: XGBoost classifier with hyperparameter optimization
- **Model Registry**: MLflow-based model versioning and promotion workflow
- **API Service**: FastAPI REST API for real-time predictions
- **Web Interface**: Streamlit application for interactive loan prediction
- **CI/CD**: Automated testing, model promotion, and Docker image building

The system predicts whether a loan application should be **Approved** or **Rejected** based on applicant information, financial details, credit history, and asset holdings.

---

## ✨ Features

### Core Functionality
- **Real-time Loan Prediction**: Instant approval/rejection predictions with confidence scores
- **Comprehensive Input Validation**: Pydantic-based schema validation for all inputs
- **Feature Engineering**: Automated preprocessing including Box-Cox transformations, outlier treatment, and feature selection
- **Model Versioning**: MLflow-based model registry with champion/challenger promotion workflow
- **Health Monitoring**: API health checks and model artifact status verification

### MLOps Capabilities
- **Reproducible Pipelines**: DVC-managed data and model pipelines
- **Experiment Tracking**: MLflow integration for metrics, parameters, and artifacts
- **Automated Model Promotion**: Intelligent promotion based on performance metrics
- **Containerized Deployment**: Docker images for both API and frontend services
- **CI/CD Integration**: GitHub Actions for automated testing and deployment

### User Experience
- **Intuitive Web Interface**: Clean, modern Streamlit UI with feature guides
- **Interactive Forms**: Real-time input validation and helpful tooltips
- **Visual Feedback**: Color-coded approval/rejection results with confidence scores
- **Feature Documentation**: Built-in guide explaining each input feature

---

## 📁 Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yaml                 # CI/CD pipeline configuration
├── data/
│   ├── raw/                        # Raw data files (DVC managed)
│   ├── interim/                    # Cleaned data after ingestion
│   ├── processed/                  # Train/test splits
│   └── features/                    # Feature-engineered datasets
├── models/
│   ├── artifacts/                    # Preprocessing artifacts (scalers, encodings, etc.)
│   │   ├── scaler.pkl
│   │   ├── boxcox_lambdas.json
│   │   ├── encodings.json
│   │   └── selected_features.json
│   └── xgboost_model.pkl           # Trained model (local, MLflow for deployment)
├── notebooks/
│   └── Jupyter Notebook_Loan Status Prediction_ Tarun Kumar Behera.ipynb
├── reports/
│   ├── metrics.json                # Model evaluation metrics
│   └── experiment_info.json        # MLflow experiment metadata
├── scripts/
│   ├── model_promotion.py          # Model promotion workflow
│   └── model_get.py                # Model retrieval utilities
├── src/
│   ├── api/                        # FastAPI backend
│   │   ├── app.py                  # FastAPI application
│   │   ├── config.py               # API configuration
│   │   ├── predict.py               # Prediction service (MLflow integration)
│   │   └── schemas.py               # Pydantic request/response models
│   ├── data/                       # Data pipeline
│   │   ├── data_ingestion.py       # DVC data fetching and initial cleaning
│   │   └── data_preprocessing.py   # Winsorization, Box-Cox, encoding
│   ├── features/                   # Feature engineering
│   │   └── feature_engineering.py  # SMOTE, scaling, feature selection
│   ├── model/                      # Model training and evaluation
│   │   ├── model_building.py       # XGBoost model training
│   │   ├── model_evaluation.py     # Model evaluation and MLflow logging
│   │   └── model_registration.py   # MLflow model registry integration
│   ├── streamlit_app/              # Streamlit frontend
│   │   ├── app.py                  # Main Streamlit application
│   │   └── utils.py                # API client and feature definitions
│   └── config.py                   # Shared configuration
├── tests/                          # Integration tests
│   ├── test_api.py
│   ├── test_datapipeline.py
│   └── test_model.py
├── docker-compose.yml              # Local Docker orchestration
├── Dockerfile.api                  # FastAPI Docker image
├── Dockerfile.streamlit            # Streamlit Docker image
├── dvc.yaml                        # DVC pipeline definition
├── params.yaml                     # Pipeline configuration parameters
├── requirements.txt                # Full project dependencies
├── requirements-api.txt            # API service dependencies
├── requirements-streamlit.txt      # Streamlit app dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────┐
│   Streamlit UI  │  (Frontend - Port 8501)
│   (User Input)  │
└────────┬────────┘
         │ HTTP POST /predict
         ▼
┌─────────────────┐
│   FastAPI API   │  (Backend - Port 8000)
│  (Validation)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LoanPredictor  │
│  (Preprocessing) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MLflow Registry │  (Model & Artifacts)
│  (XGBoost Model) │
└─────────────────┘
```

### Component Details

#### 1. FastAPI Backend (`src/api/`)

**Endpoints:**
- `GET /` - Service information and status
- `GET /health` - Health check with model artifact status
- `POST /predict` - Loan prediction endpoint

**Key Features:**
- CORS middleware for cross-origin requests
- Pydantic validation for request/response schemas
- MLflow integration for model loading
- Automatic artifact loading on startup
- Health check endpoint for deployment monitoring

**Configuration:**
- Model name: `loan-approval-xgboost`
- MLflow tracking URI: Configurable via environment variables
- Model alias: `champion` (active model)

#### 2. Prediction Service (`src/api/predict.py`)

**Preprocessing Pipeline:**
1. **Box-Cox Transformation**: Applied to skewed monetary features using saved lambda parameters
2. **Feature Engineering**: Creates `total_assets` from sum of transformed asset values
3. **Categorical Encoding**: 
   - Ordinal encoding for education level
   - One-hot encoding for employment type and self-employment status
4. **Feature Selection**: Selects 6 key features used in training
5. **Standardization**: Applies StandardScaler fitted on training data

**Model Features (in order):**
- `cibil_score` - Credit score (300-900)
- `loan_term` - Loan duration in years
- `loan_amount` - Requested loan amount (Box-Cox transformed)
- `total_assets` - Sum of all asset values (Box-Cox transformed)
- `income_annum` - Annual income
- `employment_type_Salaried` - Binary indicator for salaried employment

#### 3. Streamlit Frontend (`src/streamlit_app/`)

**Features:**
- Three-tab interface: Prediction, Feature Guide, About
- Real-time API health checking
- Input validation with helpful tooltips
- Visual feedback with color-coded results
- Application summary with key metrics
- Responsive design with modern UI

**Configuration:**
- API URL: Configurable via `API_URL` environment variable
- Default: `http://localhost:8000` (local) or `http://fastapi:8000` (Docker)

---

## 🔄 MLOps Pipeline

### DVC Pipeline Stages

The project uses DVC (Data Version Control) to manage the complete ML pipeline:

```yaml
1. data_ingestion      → Fetch data from DVC remote, initial cleaning
2. data_preprocessing → Winsorization, Box-Cox, encoding, train/test split
3. feature_engineering → SMOTE, scaling, feature selection
4. model_building     → Train XGBoost model
5. model_evaluation   → Evaluate model, log to MLflow
6. model_registration → Register model to MLflow registry as "challenger"
```

**Run the complete pipeline:**
```bash
dvc repro
```

**Run specific stage:**
```bash
dvc repro data_preprocessing
```

### MLflow Model Registry Workflow

1. **Model Registration** (`model_registration.py`)
   - Registers trained model to MLflow Model Registry
   - Assigns `challenger` alias (staging)
   - Tags model with metadata (framework, task, deployment_status)

2. **Model Promotion** (`scripts/model_promotion.py`)
   - Compares challenger metrics with current champion
   - Promotion criteria:
     - Minimum F1 score: 0.95
     - Improvement threshold: 1% F1 score improvement over champion
     - Auto-promote if no champion exists (meets minimum threshold)
   - Atomic promotion: Retire old champion → Promote new → Cleanup aliases

3. **Model Aliases**
   - `champion`: Active model (served by API)
   - `challenger`: Candidate model (for promotion testing)

**Promotion Decision Logic:**
- **Case 1**: No champion model → Auto-promote if F1 ≥ 0.95
- **Case 2**: Champion exists → Promote if F1 improvement ≥ 1%
- **Case 3**: Same version → Skip (already champion)

---

## 🔬 Model Development

### Data Pipeline

#### 1. Data Ingestion
- Fetches data from DVC remote storage (DagsHub)
- Drops identifier columns (`loan_id`)
- Saves cleaned data to `data/interim/`

#### 2. Data Preprocessing
- **Missing Value Handling**: Median imputation for numeric, mode for categorical
- **Outlier Treatment**: Class-wise Winsorization (5th-90th percentile)
  - Applied to: `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `bank_asset_value`
- **Box-Cox Transformation**: Reduces right-skewness in monetary features
  - Applied to: `loan_amount`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
  - Lambda parameters saved for inference
- **Feature Engineering**: Creates `total_assets` from sum of asset columns
- **Categorical Encoding**:
  - Education: Ordinal (8th=0, 10th=1, 12th=2, Graduate=3)
  - Employment type: One-hot (drop-first)
  - Self-employed: One-hot (drop-first)
- **Target Encoding**: Approved=1, Rejected=0
- **Train/Test Split**: 80/20 stratified split (random_state=42)

#### 3. Feature Engineering
- **SMOTE**: Synthetic Minority Oversampling on training data (random_state=42)
- **Feature Selection**: 6 features selected based on Random Forest importance
- **Standardization**: StandardScaler fitted on SMOTE training data

### Model Training

**Algorithm**: XGBoost Classifier

**Hyperparameters** (optimized via GridSearchCV):
```yaml
n_estimators: 200
max_depth: 3
learning_rate: 0.2
subsample: 0.75
colsample_bytree: 0.75
eval_metric: logloss
random_state: 42
```

**Training Process:**
1. Load SMOTE-balanced, scaled training features
2. Train XGBoost with optimized hyperparameters
3. Evaluate on test set (scaled with training scaler)
4. Log model and metrics to MLflow
5. Register model to MLflow Model Registry

**Model Performance** (typical):
- Accuracy: > 85%
- Precision: > 85%
- Recall: > 85%
- F1 Score: > 85%

---

## 📡 API Documentation

### Base URL

**Deployed**: `https://loan-api-latest.onrender.com`

**Local**: `http://localhost:8000`

### Endpoints

#### 1. GET `/`

Service information endpoint.

**Response:**
```json
{
  "message": "Loan Status Prediction API",
  "docs": "/docs",
  "health": "/health"
}
```

#### 2. GET `/health`

Health check endpoint with model artifact status.

**Response:**
```json
{
  "status": "healthy" | "unhealthy",
  "model_loaded": true | false
}
```

#### 3. POST `/predict`

Loan prediction endpoint.

**Request Body:**
```json
{
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
  "bank_asset_value": 3000000.0
}
```

**Field Constraints:**
- `no_of_dependents`: integer ≥ 0
- `education`: "8th" | "10th" | "12th" | "Graduate"
- `self_employed`: "Yes" | "No"
- `employment_type`: "Salaried" | "Business" | "Freelancer"
- `income_annum`: float > 0
- `loan_amount`: float > 0
- `loan_term`: float > 0 (years)
- `cibil_score`: float, 300 ≤ value ≤ 900
- `residential_assets_value`: float ≥ 0
- `commercial_assets_value`: float ≥ 0
- `luxury_assets_value`: float ≥ 0
- `bank_asset_value`: float ≥ 0

**Response:**
```json
{
  "status": "Approved" | "Rejected",
  "probability": 0.85,
  "model_version": "champion"
}
```

**Example cURL:**
```bash
curl -X POST https://loan-api-latest.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "bank_asset_value": 3000000.0
  }'
```

**Interactive API Documentation:**
- Swagger UI: `https://loan-api-latest.onrender.com/docs`
- ReDoc: `https://loan-api-latest.onrender.com/redoc`

---

## 💻 Local Setup

### Prerequisites

- **Python**: 3.10+ (3.13 recommended)
- **DVC**: For data version control
- **Docker** (optional): For containerized deployment
- **Git**: For version control

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Tarun304/Loan-Status-Prediction-using-Machine-Learning.git
cd Loan-Status-Prediction-using-Machine-Learning
```

2. **Create and activate virtual environment:**
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure DVC (if using remote storage):**
```bash
# Set up DVC remote (DagsHub)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
```

5. **Configure MLflow (for model registry):**
Create a `.env` file in the project root:
```env
MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/YOUR_REPO.mlflow
MLFLOW_TRACKING_USERNAME=YOUR_DAGSHUB_USERNAME
MLFLOW_TRACKING_PASSWORD=YOUR_DAGSHUB_TOKEN
DAGSHUB_USERNAME=YOUR_DAGSHUB_USERNAME
DAGSHUB_TOKEN=YOUR_DAGSHUB_TOKEN
```

6. **Pull data from DVC:**
```bash
dvc pull
```

---

## 🚀 Running the Application

### Option 1: Run Complete ML Pipeline

**Train and register a new model:**
```bash
# Run complete DVC pipeline
dvc repro

# Register model to MLflow (after evaluation)
python src/model/model_registration.py

# Promote model (if meets criteria)
python scripts/model_promotion.py
```

### Option 2: Run API Service Only

**Start FastAPI backend:**
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

**Verify API is running:**
```bash
curl http://localhost:8000/health
```

### Option 3: Run Streamlit Frontend

**Start Streamlit app:**
```bash
streamlit run src/streamlit_app/app.py
```

The app will open at `http://localhost:8501`

**Note:** Ensure the API is running before starting Streamlit, or update `API_URL` in `src/streamlit_app/utils.py` to point to the deployed API.

### Option 4: Docker Compose (Both Services)

**Build and run with Docker Compose:**
```bash
# Build images
docker-compose build

# Start services
docker-compose up

# Run in background
docker-compose up -d
```

Services will be available at:
- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

**Stop services:**
```bash
docker-compose down
```

### Option 5: Individual Docker Containers

**Build API image:**
```bash
docker build -f Dockerfile.api -t loan-api:latest .
```

**Run API container:**
```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=YOUR_MLFLOW_URI \
  -e MLFLOW_TRACKING_USERNAME=YOUR_USERNAME \
  -e MLFLOW_TRACKING_PASSWORD=YOUR_TOKEN \
  loan-api:latest
```

**Build Streamlit image:**
```bash
docker build -f Dockerfile.streamlit -t loan-streamlit:latest .
```

**Run Streamlit container:**
```bash
docker run -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000 \
  loan-streamlit:latest
```

---

## 🌐 Deployment

### Render Deployment

The project is configured for deployment on Render.com:

#### API Service Deployment

1. **Create a new Web Service** on Render
2. **Connect your GitHub repository**
3. **Configure settings:**
   - **Build Command**: `docker build -f Dockerfile.api -t loan-api .`
   - **Start Command**: `docker run -p $PORT:8000 loan-api`
   - **Environment Variables**:
     ```
     MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
     MLFLOW_TRACKING_USERNAME=your_username
     MLFLOW_TRACKING_PASSWORD=your_token
     DAGSHUB_USERNAME=your_username
     DAGSHUB_TOKEN=your_token
     ```

#### Streamlit Service Deployment

1. **Create a new Web Service** on Render
2. **Connect your GitHub repository**
3. **Configure settings:**
   - **Build Command**: `docker build -f Dockerfile.streamlit -t loan-streamlit .`
   - **Start Command**: `docker run -p $PORT:8501 loan-streamlit`
   - **Environment Variables**:
     ```
     API_URL=https://loan-api-latest.onrender.com
     ```

### Alternative Deployment Options

- **AWS**: Use ECS/Fargate with Application Load Balancer
- **Google Cloud**: Cloud Run for serverless deployment
- **Azure**: Container Instances or App Service
- **Heroku**: Container registry deployment
- **Kubernetes**: For scalable, large-scale deployment

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.13**: Programming language
- **XGBoost 3.0.5**: Gradient boosting classifier
- **FastAPI 0.116.2**: Modern, fast web framework for APIs
- **Streamlit 1.49.1**: Rapid web app development
- **Pydantic 2.11.9**: Data validation using Python type annotations

### Data Science & ML
- **Pandas 2.3.2**: Data manipulation and analysis
- **NumPy 2.3.3**: Numerical computing
- **Scikit-learn 1.7.2**: Machine learning utilities
- **imbalanced-learn 0.14.1**: SMOTE for class balancing
- **SciPy 1.16.2**: Statistical functions (Box-Cox, Winsorization)

### MLOps & DevOps
- **MLflow 3.8.1**: Experiment tracking and model registry
- **DVC 3.65.0**: Data version control and pipeline management
- **Docker**: Containerization
- **GitHub Actions**: CI/CD automation

### Infrastructure
- **Uvicorn 0.35.0**: ASGI server for FastAPI
- **DagsHub**: MLflow tracking server and DVC remote storage
- **Render**: Cloud hosting platform

---

## 🧪 Testing

### Run Tests

**All tests:**
```bash
pytest tests/ -v
```

**Specific test suites:**
```bash
# Model tests
pytest tests/test_model.py -v

# API tests
pytest tests/test_api.py -v

# Data pipeline tests
pytest tests/test_datapipeline.py -v
```

**With coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

- **`test_model.py`**: Model training and evaluation tests
- **`test_api.py`**: FastAPI endpoint tests
- **`test_datapipeline.py`**: Data preprocessing and feature engineering tests

---

## 🔄 CI/CD Pipeline

The project includes a comprehensive GitHub Actions workflow (`.github/workflows/ci.yaml`) that:

### Pipeline Stages

1. **ML Pipeline Execution**
   - Checks out code
   - Sets up Python 3.13 environment
   - Installs dependencies using UV
   - Configures DVC and pulls data
   - Runs complete DVC pipeline (`dvc repro`)
   - Executes unit tests
   - Pushes results to DVC remote

2. **Model Promotion**
   - Runs model promotion script
   - Compares challenger with champion
   - Promotes if criteria met
   - Creates promotion flag artifact

3. **Docker Image Building**
   - Checks which services changed
   - Builds FastAPI image if API files changed or model promoted
   - Builds Streamlit image if frontend files changed or model promoted
   - Pushes images to GitHub Container Registry
   - Uses build caching for faster builds

### Trigger Conditions

- **Push to main branch**: Full pipeline execution
- **Pull requests**: Pipeline validation (no deployment)
- **Model promotion**: Automatic Docker image rebuild

---

## 📊 Model Performance

### Key Metrics

The model achieves strong performance on the test set:

- **Accuracy**: > 85%
- **Precision**: > 85%
- **Recall**: > 85%
- **F1 Score**: > 85%

### Feature Importance

The model uses 6 carefully selected features:

1. **CIBIL Score**: Strongest predictor of loan approval
2. **Loan Term**: Duration of loan repayment
3. **Loan Amount**: Requested loan amount (Box-Cox transformed)
4. **Total Assets**: Aggregated asset value (Box-Cox transformed)
5. **Annual Income**: Applicant's yearly income
6. **Employment Type (Salaried)**: Binary indicator for salaried employment

### Model Limitations

- Trained on historical data - may not reflect current market conditions
- Does not account for external economic factors
- Binary classification only (Approved/Rejected) - no risk scoring
- Requires all input features to be provided

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Tarun Kumar Behera**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 👤 Author

**Tarun Kumar Behera**

- GitHub: [@Tarun304](https://github.com/Tarun304)
- Project Repository: [Loan-Status-Prediction-using-Machine-Learning](https://github.com/Tarun304/Loan-Status-Prediction-using-Machine-Learning)

---

## 🙏 Acknowledgments

- **DagsHub** for MLflow and DVC hosting
- **Render** for free tier hosting
- **XGBoost** team for the excellent gradient boosting library
- **FastAPI** and **Streamlit** communities for great documentation

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. The loan approval predictions are based on a machine learning model trained on historical data and should **not** be used as the sole basis for actual loan approval decisions. Real-world loan approval processes involve many additional factors including regulatory requirements, policy changes, and individual circumstances that are not captured in this model.


