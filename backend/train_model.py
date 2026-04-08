import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

def create_synthetic_data(num_samples=2000):
    np.random.seed(42)
    # Generate random features
    age = np.random.randint(21, 65, num_samples)
    income = np.random.randint(30000, 150000, num_samples)
    loan_amount = np.random.randint(1000, 50000, num_samples)
    credit_score = np.random.randint(300, 850, num_samples)
    employment_years = np.random.randint(0, 40, num_samples)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'EmploymentYears': employment_years
    })
    
    # Calculate a simplified risk score
    # Lower income, higher loan amount, lower credit score -> higher risk (closer to 1)
    risk_score = (
        ((50000 / df['Income']) * 0.4) + 
        ((df['LoanAmount'] / 20000) * 0.3) + 
        ((700 / df['CreditScore']) * 0.4) - 
        (df['EmploymentYears'] * 0.01)
    )
    
    # Add some noise
    risk_score += np.random.normal(0, 0.2, num_samples)
    
    # 1 is Default (Reject), 0 is Good (Approve)
    # Let's say if risk_score > 1.2, they default.
    df['Default'] = (risk_score > 1.2).astype(int)
    
    return df

def train_and_save_model():
    print("Generating synthetic dataset...")
    df = create_synthetic_data(3000)
    
    X = df[['Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentYears']]
    y = df['Default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model test accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'loan_xgboost_model.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
