import pandas as pd
import joblib
import numpy as np
from pathlib import Path

class LoanPredictor:
    def __init__(self):
        models_dir = Path(__file__).parent.parent.parent / "models"
        self.lambdas = joblib.load(models_dir / 'lambdas.joblib')
        self.scaler = joblib.load(models_dir / 'scaler.joblib') 
        self.model = joblib.load(models_dir / 'xgboost_model.pkl')
    
    def boxcox_transform(self, value, lambda_value):
        return (value ** lambda_value - 1) / lambda_value if lambda_value != 0 else np.log(value)
    
    def predict(self, input_data):
        try:
            df = pd.DataFrame([input_data])
            
            # Apply transformations
            df['loan_amount'] = self.boxcox_transform(df['loan_amount'] + 1, self.lambdas['loan_amount'])
            df['residential_assets_value'] = self.boxcox_transform(df['residential_assets_value'] + 1, self.lambdas['residential_assets_value'])
            df['commercial_assets_value'] = self.boxcox_transform(df['commercial_assets_value'] + 1, self.lambdas['commercial_assets_value'])
            df['luxury_assets_value'] = self.boxcox_transform(df['luxury_assets_value'] + 1, self.lambdas['luxury_assets_value'])
            df['bank_asset_value'] = self.boxcox_transform(df['bank_asset_value'] + 1, self.lambdas['bank_asset_value'])
            
            df['total_assets'] = (df['residential_assets_value'] + df['commercial_assets_value'] + 
                                 df['luxury_assets_value'] + df['bank_asset_value'])
            
            df['employment_type_Salaried'] = df['employment_type'].apply(lambda x: 1 if x == 'Salaried' else 0)
            
            df = df[['cibil_score', 'loan_term', 'loan_amount', 'total_assets', 'income_annum', 'employment_type_Salaried']]
            
            scaled_features = self.scaler.transform(df)
            prediction = self.model.predict(scaled_features)[0]
            probability = self.model.predict_proba(scaled_features)[0]
            
            return {
                "status": "Approved" if prediction == 1 else "Rejected",
                "probability": float(max(probability))
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {
                "status": "Error",
                "probability": 0.0,
                "error": str(e)
            }
