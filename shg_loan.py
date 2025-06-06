import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def ensure_model_directory():
    """Ensure the models directory exists"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created models directory")

def train_model():
    """Train and save the SHG loan eligibility model using existing dataset"""
    ensure_model_directory()
    
    # Check if dataset exists
    dataset_path = 'models/shg_loan_dataset_balanced.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} records")
    
    # Check column names
    print("Dataset columns:", df.columns.tolist())
    
    # Preprocessing
    X = df.drop('Eligibility', axis=1)
    y = df['Eligibility']
    
    # Create label encoders for each column
    le_dict = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
    
    # Encode target variable
    label_target = LabelEncoder()
    y = label_target.fit_transform(y)
    
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
    with open('models/shg_loan_model.pkl', 'wb') as f:
        pickle.dump((model, le_dict, label_target), f)
    
    # Save feature names
    with open('models/shg_loan_features_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print("Model saved successfully")
    return model, le_dict, label_target

def predict_eligibility(input_data):
    """
    Predict SHG loan eligibility based on input data
    
    Args:
        input_data (dict): Dictionary containing user input data
        
    Returns:
        tuple: (eligibility (str), probability (float), explanation (str))
    """
    ensure_model_directory()
    
    # Map form field names to model field names
    field_mapping = {
        'formation_year': 'Formation Year',
        'member_count': 'Member Count',
        'loan_amount': 'Loan Amount',
        'loan_provider_bank': 'Loan Provider Bank',
        'annual_income': 'Group Annual Income',
        'loan_usage': 'Loan Usage'
    }
    
    # Create a new dictionary with mapped field names
    mapped_data = {}
    for form_field, model_field in field_mapping.items():
        if form_field in input_data:
            mapped_data[model_field] = input_data[form_field]
    
    # Print the mapped data for debugging
    print("Mapped input data:", mapped_data)
    
    # Check if we have all required fields
    required_fields = ['Formation Year', 'Member Count', 'Loan Amount', 
                       'Loan Provider Bank', 'Group Annual Income', 'Loan Usage']
    
    missing_fields = [field for field in required_fields if field not in mapped_data]
    if missing_fields:
        print(f"Missing required fields: {missing_fields}")
        return rule_based_prediction(mapped_data)
    
    try:
        # Check if model exists, if not train it
        model_path = 'models/shg_loan_model.pkl'
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, training new model...")
            train_model()
        
        # Load model and encoders
        with open(model_path, 'rb') as f:
            model, le_dict, label_target = pickle.load(f)
        
        # Load feature names if available
        feature_names = None
        feature_path = 'models/shg_loan_features_names.pkl'
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                feature_names = pickle.load(f)
                print("Loaded feature names:", feature_names)
        
        # Prepare input data
        input_df = pd.DataFrame([mapped_data])
        
        # Encode input data
        for col in input_df.columns:
            if col in le_dict:
                try:
                    input_df[col] = le_dict[col].transform(input_df[col])
                except ValueError:
                    # Handle unknown categories
                    print(f"Unknown category in column {col}: {input_df[col][0]}")
                    # Use the most frequent category as fallback
                    input_df[col] = 0
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        probability = probabilities[prediction]
        
        # Convert prediction back to label
        eligibility = label_target.inverse_transform([prediction])[0]
        
        # Generate explanation
        explanation = generate_explanation(mapped_data, eligibility, probability)
        
        return eligibility, probability, explanation
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fall back to rule-based prediction
        return rule_based_prediction(mapped_data)

def rule_based_prediction(input_data):
    """Fallback rule-based prediction when model fails"""
    print("Using rule-based prediction as fallback...")
    
    # Initialize variables
    is_eligible = False
    probability = 0.5
    explanation_points = []
    
    # Rule 1: Formation year 2022-2024 increases eligibility
    if 'Formation Year' in input_data:
        try:
            formation_year = int(input_data['Formation Year'])
            if formation_year in [2022, 2023, 2024]:
                is_eligible = True
                probability += 0.1
                explanation_points.append("- Your SHG was formed recently (2022-2024), which is favorable.")
            else:
                probability -= 0.1
                explanation_points.append("- Your SHG formation year is outside the preferred range.")
        except (ValueError, TypeError):
            pass
    
    # Rule 2: Member count between 10-15 increases eligibility
    if 'Member Count' in input_data:
        if input_data['Member Count'] in ['Less than 10', '10 to 15']:
            is_eligible = True
            probability += 0.1
            explanation_points.append("- Your SHG member count is within the preferred range.")
        else:
            probability -= 0.1
            explanation_points.append("- Your SHG has too many members, which reduces eligibility.")
    
    # Rule 3: Loan amount less than 2 Lakh increases eligibility
    if 'Loan Amount' in input_data:
        if input_data['Loan Amount'] in ['less than 1Lakh', '1Lakh to 2Lakh', 'below 1Lakh', '1Lakh to 2Lakh']:
            is_eligible = True
            probability += 0.1
            explanation_points.append("- Your loan amount is within the eligible range.")
        else:
            probability -= 0.2
            explanation_points.append("- Your loan amount exceeds the maximum threshold for this waiver.")
    
    # Rule 4: Eligible banks increase eligibility
    if 'Loan Provider Bank' in input_data:
        eligible_banks = ['Indian Bank', 'IOB', 'Canara Bank', 'Co-operation Bank', 'Grama Bank', 'Cooperative Bank']
        if input_data['Loan Provider Bank'] in eligible_banks:
            is_eligible = True
            probability += 0.1
            explanation_points.append("- Your loan provider bank is eligible for this waiver scheme.")
        else:
            probability -= 0.2
            explanation_points.append("- Your loan provider bank is not part of the eligible institutions.")
    
    # Rule 5: Low income increases eligibility
    if 'Group Annual Income' in input_data:
        if input_data['Group Annual Income'] in ['less than ₹90K', 'below 90K']:
            is_eligible = True
            probability += 0.1
            explanation_points.append("- Your group's annual income qualifies for assistance.")
        else:
            probability -= 0.2
            explanation_points.append("- Your group's annual income exceeds the threshold for this waiver.")
    
    # Rule 6: Productive loan usage increases eligibility
    if 'Loan Usage' in input_data:
        if input_data['Loan Usage'] in ['Livelihood', 'Productive Asset']:
            is_eligible = True
            probability += 0.1
            explanation_points.append("- Your loan was used for productive purposes, which is favorable.")
        else:
            probability -= 0.1
            explanation_points.append("- The purpose of your loan reduces eligibility.")
    
    # Cap probability between 0.1 and 0.9
    probability = max(0.1, min(0.9, probability))
    
    # Final decision based on probability
    is_eligible = probability >= 0.5
    eligibility = "Eligible" if is_eligible else "Not Eligible"
    
    # Generate explanation
    if is_eligible:
        explanation = f"Your SHG is eligible for the loan waiver with {probability:.1%} confidence.\n" + "\n".join(explanation_points)
    else:
        explanation = f"Your SHG is not eligible for the loan waiver with {(1-probability):.1%} confidence.\n" + "\n".join(explanation_points)
    
    return eligibility, probability, explanation

def generate_explanation(input_data, eligibility, probability):
    """Generate human-readable explanation for the prediction"""
    explanation_points = []
    
    if eligibility == "Eligible":
        explanation = f"Your SHG is eligible for the loan waiver with {probability:.1%} confidence.\n"
        
        # Add specific reasons
        if input_data.get('Formation Year') in [2022, 2023, 2024] or str(input_data.get('Formation Year')) in ['2022', '2023', '2024']:
            explanation_points.append("- Your SHG was formed recently (2022-2024), which is favorable.")
            
        if input_data.get('Member Count') in ['Less than 10', '10 to 15']:
            explanation_points.append("- Your SHG member count is within the preferred range.")
            
        if input_data.get('Loan Amount') in ['less than 1Lakh', '1Lakh to 2Lakh', 'below 1Lakh']:
            explanation_points.append("- Your loan amount is within the eligible range.")
            
        eligible_banks = ['Indian Bank', 'IOB', 'Canara Bank', 'Co-operation Bank', 'Grama Bank', 'Cooperative Bank']
        if input_data.get('Loan Provider Bank') in eligible_banks:
            explanation_points.append("- Your loan provider bank is eligible for this waiver scheme.")
            
        if input_data.get('Group Annual Income') in ['less than ₹90K', 'below 90K']:
            explanation_points.append("- Your group's annual income qualifies for assistance.")
            
        if input_data.get('Loan Usage') in ['Livelihood', 'Productive Asset']:
            explanation_points.append("- Your loan was used for productive purposes, which is favorable.")
    else:
        explanation = f"Your SHG is not eligible for the loan waiver with {(1-probability):.1%} confidence.\n"
        
        # Add specific reasons for rejection
        if input_data.get('Formation Year') not in [2022, 2023, 2024] and str(input_data.get('Formation Year')) not in ['2022', '2023', '2024']:
            explanation_points.append("- Your SHG formation year is outside the preferred range.")
            
        if input_data.get('Member Count') == 'Greater than 15':
            explanation_points.append("- Your SHG has too many members, which reduces eligibility.")
            
        if input_data.get('Loan Amount') in ['above 2Lakh']:
            explanation_points.append("- Your loan amount exceeds the maximum threshold for this waiver.")
            
        eligible_banks = ['Indian Bank', 'IOB', 'Canara Bank', 'Co-operation Bank', 'Grama Bank', 'Cooperative Bank']
        if input_data.get('Loan Provider Bank') not in eligible_banks:
            explanation_points.append("- Your loan provider bank is not part of the eligible institutions.")
            
        if input_data.get('Group Annual Income') not in ['less than ₹90K', 'below 90K']:
            explanation_points.append("- Your group's annual income exceeds the threshold for this waiver.")
            
        if input_data.get('Loan Usage') == 'Other':
            explanation_points.append("- The purpose of your loan reduces eligibility.")
    
    # Add explanation points to the main explanation
    explanation += "\n".join(explanation_points)
    
    return explanation

# If run directly, train the model
if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
