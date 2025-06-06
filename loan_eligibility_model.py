import pandas as pd
import random
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def generate_dataset():
    # Define options
    loan_amount = ['less than 50k','60k-1Lakh','1Lakh -2Lakh','<2Lakh']
    repayment_option = ['Fully Paid', 'Partially Paid', 'Default']
    land_size = ['1-2', '2-4', '5-6']
    collateral_type = ['Gold', 'Land', 'Vehicle']
    collateral_value = ['<50K', '60K-1Lakh', '>1Lakh']
    subsidy_option = ['Yes', 'No']
    income_level = ['less than 70k','80k-1Lakh','above 1.5Lakh']
    card_types = ['PHH', 'PHH-AYY', 'NPHH', 'None']
    loan_history = ['No Loan', '1-2 Loans', '>2 Loans']
    loan_purpose = ['Crop', 'Livestock', 'Machinery', 'Irrigation']
    interest_rate = ['less than 4','5','6','7']
    loan_tenure = ['less than 3 yrs','above 3 yrs']

    # Define eligibility rules
    def determine_eligibility(row):
        if row['Subsidy'] == 'Yes':
            return 'Eligible'
        if row['Income'] == 'less than 70k' and row['Card_Type'] == 'PHH':
            return 'Eligible'
        if row['Income'] == 'less than 70k' and row['Card_Type'] == 'PHH-AYY':
            return 'Eligible'
        if row['Income'] == 'less than 70k' and row['Card_Type'] == 'NPHH' and row['Land_Size'] in ('1-2','2-4'):
            return 'Eligible'
        if row['Income'] == 'less than 70k' and row['Card_Type'] == 'NPHH' and row['Land_Size'] == '5-6':
            return 'Not Eligible'
        if row['Collateral_Type'] in ('Gold','Land') and row['Land_Size'] == '2-4' and row['Loan_Tenure'] == 'less than 3 yrs':
            return 'Eligible'
        if row['Collateral_Type'] in ('Gold','Land') and row['Land_Size'] == '2-4' and row['Loan_Tenure'] == 'above 3 yrs':
            return 'Not Eligible'
        if row['Loan_Tenure'] == 'less than 3 yrs':
            return 'Eligible'
        if row['Loan_Tenure'] == 'above 3 yrs':
            return 'Not Eligible'
        if row['Collateral_Type'] == 'Gold' and row['Repayment_Status'] == 'Default':
            return 'Eligible'
        if row['Repayment_Status'] == 'Fully Paid':
            return 'Not Eligible'
        if row['Loan_Amount'] =='<2Lakh':
            return 'Not Eligible'
        if row['Loan_Tenure'] == 'less than 3 yrs':
            return 'Eligible'
        if row['Loan_Tenure'] == 'above 3 yrs':
            return 'Not Eligible'
        if row['Land_Size'] == '5-6' and row['Income'] == 'above 1.5Lakh':
            return 'Not Eligible'
        if row['Collateral_Value'] == '>1Lakh' and row['Repayment_Status'] == 'Fully Paid':
            return 'Not Eligible'
        if row['Collateral_Type'] == 'Vehicle' and row['Repayment_Status'] == 'Partially Paid' and row['Collateral_Value'] == '<50K':
            return 'Not Eligible'
        if row['Collateral_Type'] == 'Vehicle' and row['Repayment_Status'] == 'Default' and row['Collateral_Value'] == '<50K':
            return 'Not Eligible'
        if row['Income'] == 'above 1.5Lakh': 
            return 'Not Eligible'
        return 'Eligible'

    # Generate balanced dataset
    eligible_data = []
    non_eligible_data = []

    while len(eligible_data) < 100 or len(non_eligible_data) < 100:
        row = {
            'Loan_Amount': random.choice(loan_amount),
            'Land_Size': random.choice(land_size),
            'Repayment_Status': random.choice(repayment_option),
            'Loan_Tenure': random.choice(loan_tenure),
            'Interest_Rate': random.choice(interest_rate),
            'Collateral_Type': random.choice(collateral_type),
            'Collateral_Value': random.choice(collateral_value),
            'Subsidy': random.choice(subsidy_option),
            'Income': random.choice(income_level),
            'Card_Type': random.choice(card_types),
            'Previous_Loan_History': random.choice(loan_history),
            'Loan_Purpose': random.choice(loan_purpose)
        }
        row['Eligibility'] = determine_eligibility(row)
        
        if row['Eligibility'] == 'Eligible' and len(eligible_data) < 100:
            eligible_data.append(row)
        elif row['Eligibility'] == 'Not Eligible' and len(non_eligible_data) < 100:
            non_eligible_data.append(row)

    # Combine and save
    final_df = pd.DataFrame(eligible_data + non_eligible_data).sample(frac=1).reset_index(drop=True)
    final_df.to_csv('balanced_agri_loan_waiver_dataset_200.csv', index=False)
    
    return final_df

def train_model():
    # Check if dataset exists, if not generate it
    if not os.path.exists('balanced_agri_loan_waiver_dataset_200.csv'):
        df = generate_dataset()
    else:
        df = pd.read_csv('balanced_agri_loan_waiver_dataset_200.csv')
    
    # Prepare data for training
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(df.drop('Eligibility', axis=1))
    y = df['Eligibility'].map({'Eligible': 1, 'Not Eligible': 0})
    
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
    
    # Save model and feature names
    with open('loan_eligibility_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names for later use
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    return model, list(X.columns)

def predict_eligibility(input_data):
    """
    Predict loan waiver eligibility based on input data
    
    Args:
        input_data (dict): Dictionary containing user input data
        
    Returns:
        tuple: (eligibility (bool), probability (float), explanation (str))
    """
    # Load model and feature names
    if not os.path.exists('loan_eligibility_model.pkl'):
        model, feature_names = train_model()
    else:
        with open('loan_eligibility_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Convert to dummy variables
    input_encoded = pd.get_dummies(input_df)
    
    # Align input data with training data columns
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_names]
    
    # Make prediction
    probability = model.predict_proba(input_encoded)[0][1]
    eligibility = probability >= 0.5
    
    # Generate explanation
    explanation = generate_explanation(input_data, eligibility, probability)
    
    return eligibility, probability, explanation

def generate_explanation(input_data, eligibility, probability):
    """Generate human-readable explanation for the prediction"""
    explanation = []
    
    if eligibility:
        explanation.append(f"You are eligible for the loan waiver with {probability:.1%} confidence.")
        
        # Add specific reasons
        if input_data.get('Subsidy') == 'Yes':
            explanation.append("- You are already receiving a subsidy, which qualifies you for the waiver.")
            
        if input_data.get('Income') == 'less than 70k':
            explanation.append("- Your income level (below ₹70,000) qualifies you for assistance.")
            
        if input_data.get('Card_Type') in ['PHH', 'PHH-AYY']:
            explanation.append("- Your ration card type (Priority Household) makes you eligible.")
            
        if input_data.get('Loan_Tenure') == 'less than 3 yrs':
            explanation.append("- Your loan tenure (less than 3 years) meets the eligibility criteria.")
            
        if input_data.get('Collateral_Type') == 'Gold' and input_data.get('Repayment_Status') == 'Default':
            explanation.append("- Your gold collateral with default status qualifies for the waiver program.")
    else:
        explanation.append(f"You are not eligible for the loan waiver with {(1-probability):.1%} confidence.")
        
        # Add specific reasons for rejection
        if input_data.get('Income') == 'above 1.5Lakh':
            explanation.append("- Your income exceeds the maximum threshold (₹1.5 Lakh) for this waiver.")
            
        if input_data.get('Loan_Amount') == '<2Lakh':
            explanation.append("- Your loan amount exceeds the maximum covered under this scheme.")
            
        if input_data.get('Loan_Tenure') == 'above 3 yrs':
            explanation.append("- Your loan tenure (above 3 years) exceeds the eligible period.")
            
        if input_data.get('Repayment_Status') == 'Fully Paid':
            explanation.append("- Your loan is already fully paid, so no waiver is applicable.")
            
        if input_data.get('Land_Size') == '5-6' and input_data.get('Income') == 'above 1.5Lakh':
            explanation.append("- Your land size and income level combined make you ineligible.")
    
    return "\n".join(explanation)

# If run directly, train the model
if __name__ == "__main__":
    train_model()