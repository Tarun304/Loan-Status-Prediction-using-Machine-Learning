"""Test model loading  with correct source"""

import mlflow
import mlflow.xgboost

mlflow.set_tracking_uri(
    "https://dagshub.com/tkbehera304/Loan-Status-Prediction-using-Machine-Learning.mlflow"
)

print("=" * 70)
print("Testing Model Loading - Version 24")
print("=" * 70)

# Test 1: Load by alias
print("\n[Test 1] Loading by alias: models:/loan-approval-xgboost@challenger")
try:
    model = mlflow.xgboost.load_model("models:/loan-approval-xgboost@challenger")
    print("✓ SUCCESS!")
    print(f"  Model type: {type(model)}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: Load by version
print("\n[Test 2] Loading by version: models:/loan-approval-xgboost/24")
try:
    model = mlflow.xgboost.load_model("models:/loan-approval-xgboost/24")
    print("✓ SUCCESS!")
    print(f"  Model type: {type(model)}")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 70)
