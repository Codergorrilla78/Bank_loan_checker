from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import os

app = FastAPI(title="Loan Application Checker API")

# Setup CORS to allow requests from the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production, e.g., ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = xgb.XGBClassifier()
model_path = os.path.join(os.path.dirname(__file__), 'loan_xgboost_model.json')

# We'll load the model upon startup or handle it dynamically
try:
    if os.path.exists(model_path):
        model.load_model(model_path)
        MODEL_LOADED = True
    else:
        MODEL_LOADED = False
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading model: {e}")

class LoanApplication(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    EmploymentYears: int

@app.post("/predict")
def predict_loan(application: LoanApplication):
    if not MODEL_LOADED:
        # Try loading again just in case it was generated later
        if os.path.exists(model_path):
            model.load_model(model_path)
        else:
            raise HTTPException(status_code=500, detail="Model not found. Run train_model.py first.")
    
    # Create a DataFrame from the input features
    features = pd.DataFrame([application.model_dump()])
    
    # Predict (Output is 0 for Approve, 1 for Default/Reject)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # The probability of class 1 (Default) and class 0 (Approve)
    prob_reject = float(probabilities[1])
    prob_approve = float(probabilities[0])
    
    return {
        "prediction_class": int(prediction),
        "status": "Rejected" if prediction == 1 else "Approved",
        "confidence_approve": round(prob_approve * 100, 2),
        "confidence_reject": round(prob_reject * 100, 2)
    }

frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
