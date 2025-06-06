# shg_model.py

import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Set seed for reproducibility
random.seed(42)

# Define categorical options
formation_years = [2021, 2022, 2023, 2024]
member_counts = ['Less than 10', '10 to 15', 'Greater than 15']
loan_amounts = ['below 1Lakh', '1Lakh to 2Lakh', 'above 2Lakh']
loan_provider_banks = ['Indian Bank', 'IOB', 'Canara Bank', 'Cooperative Bank', 'Grama Bank', 'Axis Bank', 'Others']
annual_incomes = ['below 90K', 'above 90K']
loan_usage = ['Livelihood', 'Productive Asset', 'Other']

# Generate synthetic dataset
data = []

# Eligible data
while len(data) < 100:
    record = [
        random.choice([2022, 2023, 2024]),
        random.choice(['Less than 10', '10 to 15']),
        random.choice(['below 1Lakh', '1Lakh to 2Lakh']),
        random.choice(['Indian Bank', 'IOB', 'Canara Bank', 'Cooperative Bank', 'Grama Bank']),
        'below 90K',
        random.choice(['Livelihood', 'Productive Asset']),
        'Eligible'
    ]
    data.append(record)

# Not eligible data
while len(data) < 200:
    fy = random.choice(formation_years)
    mc = random.choice(member_counts)
    la = random.choice(loan_amounts)
    lp = random.choice(loan_provider_banks)
    ai = random.choice(annual_incomes)
    lu = random.choice(loan_usage)

    if not (mc in ['Less than 10', '10 to 15'] and
            la in ['below 1Lakh', '1Lakh to 2Lakh'] and
            lp in ['Indian Bank', 'IOB', 'Canara Bank', 'Cooperative Bank', 'Grama Bank'] and
            ai == 'below 90K' and
            fy in [2022, 2023, 2024] and
            lu in ['Livelihood', 'Productive Asset']):
        data.append([fy, mc, la, lp, ai, lu, 'Not Eligible'])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    'Formation Year', 'Member Count', 'Loan Amount',
    'Loan Provider Bank', 'Group Annual Income', 'Loan Usage', 'Eligibility'
])

# Save dataset
df.to_csv('shg_loan_dataset_balanced.csv', index=False)

# Preprocessing
X = df.drop('Eligibility', axis=1)
y = df['Eligibility']

le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

label_target = LabelEncoder()
y = label_target.fit_transform(y)  # Eligible = 1, Not Eligible = 0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ensemble model
model = VotingClassifier(estimators=[
    ('dt', DecisionTreeClassifier(random_state=0)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
    ('lr', LogisticRegression(max_iter=1000))
], voting='soft')

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
with open('shg_loan_model.pkl', 'wb') as f:
    pickle.dump((model, le_dict, label_target), f)

# Prediction function
def predict_eligibility(formation_year, member_count, loan_amount, loan_provider, annual_income, loan_usage):
    with open('shg_loan_model.pkl', 'rb') as f:
        model, le_dict, label_target = pickle.load(f)

    input_data = pd.DataFrame([{
        'Formation Year': formation_year,
        'Member Count': member_count,
        'Loan Amount': loan_amount,
        'Loan Provider Bank': loan_provider,
        'Group Annual Income': annual_income,
        'Loan Usage': loan_usage
    }])

    for col in input_data.columns:
        input_data[col] = le_dict[col].transform(input_data[col])

    prediction = model.predict(input_data)
    return label_target.inverse_transform(prediction)[0]

# Gradio Interface
if __name__ == '__main__':
    iface.launch()
