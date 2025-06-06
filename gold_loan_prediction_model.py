import pandas as pd
import random
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def gold_generate_dataset():
    # Define options
    workers = ['Farmer', 'Housewife', 'Banker', 'Government staff', 'Small Business','daily wages',]
    loan_amount = ['less than 80k','above 80k']
    purpose_of_loan = ['Medical', 'Crop Investment', 'Marriage', 'Business', 'Education']
    gold_type = ['Ornament', 'Coin', 'Bar']
    gold_weight = ['less than 20g','20g to 40g','above 40g']
    gold_purity = ['22K', '18K', '24K']
    years_of_repayment = ['1','2','3']
    card_type = ['PHH', 'NPHH', 'PHH AYY']
    provider_bank_name =['Indian Bank', 'SBI', 'Grama Bank', 'Axis bank', 'IOB','coperation Bank']
    annual_income =['below 80k','above 80k']
    subsidy =['No', 'Yes']
    already_loan_waived = ['No', 'Yes']
    pension = ['Yes', 'No']

    # Define eligibility rules
    def determine_eligibility(row):
        if (
        row['Subsidy'] == 'Yes' or
        row['Already_Loan_Waived'] == 'Yes' or
        row['Gold_Weight'] == 'above 40g' or
        row['Annual_Income'] == 'above 80k' or
        row['Provider_Bank_Name'] in ['SBI', 'Axis bank'] or
        row['Workers'].lower() in ['government staff', 'banker']
        ):
            return 'Not Eligible'

        # Eligible if these conditions are met
        if (
        row['Workers'] in ['Farmer', 'Housewife', 'Small Business', 'daily wages'] and
        row['Pension'] == 'Yes' and
        row['Gold_Weight'] in ['less than 20g', '20g to 40g'] and
        row['Gold_Purity'] in ['22K', '24K', '18K'] and
        row['Years_of_Repayment'] in ['1', '2'] and
        row['Annual_Income'] == 'below 80k' and
        row['Provider_Bank_Name'] not in ['SBI', 'Axis bank']
        ):
            return 'Eligible'

        return 'Not Eligible'

    # Generate balanced dataset
    eligible_data = []
    non_eligible_data = []

    while len(eligible_data) < 100 or len(non_eligible_data) < 100:
        row = {
            'Loan_Amount': random.choice(loan_amount),
            'Workers': random.choice(workers),
            'Purpose_of_loan': random.choice(purpose_of_loan),
            'Gold_Type': random.choice(gold_type),
            'Gold_Weight': random.choice(gold_weight),
            'Gold_Purity': random.choice(gold_purity),
            'Years_of_Repayment': random.choice(years_of_repayment),
            'Card_Type': random.choice(card_type),
            'Provider_Bank_Name': random.choice(provider_bank_name),
            'Annual_Income': random.choice(annual_income),
            'Subsidy': random.choice(subsidy),
            'Already_Loan_Waived': random.choice(already_loan_waived),
            'Pension': random.choice(pension)
        }
        row['Eligibility'] = determine_eligibility(row)
        
        if row['Eligibility'] == 'Eligible' and len(eligible_data) < 100:
            eligible_data.append(row)
        elif row['Eligibility'] == 'Not Eligible' and len(non_eligible_data) < 100:
            non_eligible_data.append(row)

    # Combine and save
    final_df = pd.DataFrame(eligible_data + non_eligible_data).sample(frac=1).reset_index(drop=True)
    final_df.to_csv('balanced_gold_loan_waiver_dataset_200.csv', index=False)
    
    return final_df

def gold_train_model():
    # Check if dataset exists, if not generate it
    if not os.path.exists('balanced_gold_loan_waiver_dataset_200.csv'):
        df = gold_generate_dataset()
    else:
        df = pd.read_csv('balanced_gold_loan_waiver_dataset_200.csv')
    
    # Prepare data for training
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(df.drop('Eligibility', axis=1))
    y = df['Eligibility'].map({'Eligible': 1, 'Not Eligible': 0})
    print(X.columns)
    
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model and feature names
    with open('models/gold_eligibility_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names for later use
    with open('models/feature_gold_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    return model, list(X.columns)

def gold_predict_eligibility(input_data):
    """
    Predict loan waiver eligibility based on input data
    
    Args:
        input_data (dict): Dictionary containing user input data
        
    Returns:
        tuple: (eligibility (bool), probability (float), explanation (str))
    """
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load model and feature names
    if not os.path.exists('models/gold_eligibility_model.pkl'):
        model, feature_gold_names = gold_train_model()
    else:
        with open('models/gold_eligibility_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/feature_gold_names.pkl', 'rb') as f:
            feature_gold_names = pickle.load(f)
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Convert to dummy variables
    input_encoded = pd.get_dummies(input_df)
    
    # Align input data with training data columns
    for col in feature_gold_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_gold_names]
    
    # Make prediction
    probability = model.predict_proba(input_encoded)[0][1]
    eligibility = probability >= 0.5
    
    # Generate explanation
    explanation = gold_generate_explanation(input_data, eligibility, probability)
    
    return eligibility, probability, explanation

def gold_generate_explanation(input_data, eligibility, probability):
    """Generate human-readable explanation for the prediction"""
    explanation = []
    
    if eligibility:
        explanation.append(f"You are eligible for the loan waiver with {probability:.1%} confidence.")
        
        # Add specific reasons
        if input_data.get('Gold_Weight') == 'above 40g':
            explanation.append("- Your gold weight (above 40g) qualifies you for the waiver.")
            
        if input_data.get('Gold_Purity') in ['22K', '24K']:
            explanation.append("- Your gold purity (22K or 24K) meets the eligibility criteria.")
        if input_data.get('Workers') in ['Farmer', 'Housewife', 'Small Business', 'daily wages']:
            explanation.append("- Your occupation (Farmer, Housewife, Small Business, daily wages) qualifies you for the waiver.")
        if input_data.get('Pension') == 'Yes':
            explanation.append("- Your pension status (Yes) meets the eligibility criteria.")
        if input_data.get('Years_of_Repayment') in [1, 2]:
            explanation.append("- Your years of repayment (1 or 2 years) meets the eligibility criteria.")
        if input_data.get('Annual_Income') == 'below 80k':
            explanation.append("- Your annual income (below 80k) meets the eligibility criteria.")
        if input_data.get('Provider_Bank_Name') not in ['SBI', 'Axis']:
            explanation.append("- Your provider bank (not SBI or Axis) meets the eligibility criteria.")
        if input_data.get('Subsidy') == 'Yes':
            explanation.append("- You are already receiving a subsidy, which qualifies you for the waiver.")
        if input_data.get('Already_Loan_Waived') == 'Yes':
            explanation.append("- You have already availed a loan waiver, which makes you ineligible.")
        if input_data.get('Workers') in ['Government staff', 'Banker']:
            explanation.append("- Your occupation (Government staff, Banker) qualifies you for the waiver.")
        if input_data.get('Loan_Amount') == 'above 80k':
            explanation.append("- Your loan amount (above 80k) exceeds the maximum covered under this scheme.")        
    return "\n".join(explanation)

# If run directly, train the model
if __name__ == "__main__":
    gold_train_model()
