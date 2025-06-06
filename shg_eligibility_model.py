# shg_eligibility_model.py

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Step 1: Dataset Generation
# Define possible values for each column
repayment_status_options = ["On-time", "Irregular", "Defaulted"]
years_of_non_payment_options = ["less than 1 year", "1–3 years", "above 3 years"]
annual_income_options = ["less than ₹80,000", "above ₹80,000"]  # Only 2 options here
loan_usage_verified_options = ["Yes", "No"]
shg_linked_to_bank_options = ["Yes", "No"]
loan_amount_options = ["less than ₹1Lakh", "₹1Lakh - ₹2Lakh", "above ₹2Lakh"]
loan_provider_bank_options = ["IOB", "Axis Bank", "SBI", "Cooperation Bank", "Grama Bank", "Indian Bank"]

# Generate synthetic SHG formation years (between 2021 and 2025)  # Fix: Use current year as the upper limit
shg_formation_years = [random.randint(2021, 2025) for _ in range(200)]

# Generate records
data = []
for i in range(200):
    repayment_status = random.choices(repayment_status_options, weights=[0.4, 0.35, 0.25], k=1)[0]
    
    if repayment_status == "Defaulted":
        years_of_non_payment = random.choices(["less than 1 year", "1–3 years"], weights=[0.5, 0.5], k=1)[0]
    elif repayment_status == "Irregular":
        years_of_non_payment = random.choices(["less than 1 year", "1–3 years", "above 3 years"], weights=[0.4, 0.3, 0.3], k=1)[0]
    else:
        years_of_non_payment = "less than 1 year"
    
    # Fix: Use weights that match the number of options (2 options need 2 weights)
    annual_income = random.choices(annual_income_options, weights=[0.6, 0.4], k=1)[0]
    shg_formation_year = shg_formation_years[i]
    loan_usage_verified = random.choices(loan_usage_verified_options, weights=[0.85, 0.15], k=1)[0]
    shg_linked_to_bank = random.choices(shg_linked_to_bank_options, weights=[0.9, 0.1], k=1)[0]
    loan_amount = random.choices(loan_amount_options, weights=[0.4, 0.4, 0.2], k=1)[0]
    loan_provider_bank = random.choice(loan_provider_bank_options)

    data.append([
        repayment_status,
        years_of_non_payment,
        annual_income,
        shg_formation_year,
        loan_usage_verified,
        shg_linked_to_bank,
        loan_amount,
        loan_provider_bank
    ])

# Create DataFrame
columns = [
    "Repayment Status",
    "Years of Non-Payment",
    "Annual Income",
    "SHG Formation Year",
    "Loan Usage Verified",
    "SHG Linked to Bank",
    "Loan Amount",
    "Loan Provider Bank"
]
df = pd.DataFrame(data, columns=columns)

# Step 2: Preprocessing
# Convert categorical data into numeric format using Label Encoding
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features (X) and Target (y)
X = df.drop("Repayment Status", axis=1)
y = df["Repayment Status"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for ensemble models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Training (Ensemble Learning - Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)

# Print Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and label encoders for later use
import joblib
joblib.dump(model, 'shg_eligibility_model.pkl')
joblib.dump(label_encoders, 'shg_eligibility_label_encoders.pkl')
joblib.dump(scaler, 'shg_eligibility_scaler.pkl')

