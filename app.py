from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import sqlite3
import os
import datetime
import re
from werkzeug.utils import secure_filename
import cv2
from pyzbar.pyzbar import decode
from qr_utils import extract_qr_data, parse_aadhaar_data, parse_smartcard_data ,save_session_data
from loan_eligibility_model import predict_eligibility
from gold_loan_prediction_model import gold_predict_eligibility
import joblib
import pandas as pd
import random
import pickle
from joblib import load
import io
import pdfkit    # pip install pdfkit, and wkhtmltopdf system‑wide

# Import the SHG model functions
from shg_loan import predict_eligibility as predict_shg_eligibility

# 1. Point to the exe
config = pdfkit.configuration(
    wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"    # ⇧  use your real location
)

# 2. Example of how to use pdfkit (commented out - this is just an example)
'''
# This is how you would use pdfkit in a route function:
html_content = "<h1>Hello World</h1>"
pdf_bytes = pdfkit.from_string(html_content, False, configuration=config)
'''

app = Flask(__name__)
app.secret_key = 'secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create models directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load the trained models - COMMENT OUT THESE LINES FOR NOW
# model = joblib.load('models/loan_eligibility_model.pkl')  
# feature_names = joblib.load('models/feature_names.pkl')  
def init_db():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        mobile TEXT,
        password TEXT,
        account_creation TIMESTAMP
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS aadhar_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        gender TEXT,
        dob TEXT,
        aadhaar_number TEXT,
        Address TEXT,
        mobile TEXT
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS bankers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        bank_id TEXT,
        designation TEXT,
        password TEXT
    )''')

    conn.commit()
    conn.close()

    con = sqlite3.connect('dashboard.db')
    cursor = con.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        account_creation TIMESTAMP,
        aadhar_image_path TEXT,
        match_result TEXT
    )''')
    con.commit()
    con.close()


init_db()


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/auth', methods=['GET', 'POST'])
def auth():
    error = None

    if request.method == 'POST':
        if 'login' in request.form:
            # Login form submitted
            mobile = request.form['login_mobile']
            password = request.form['login_password']

            conn = sqlite3.connect('user_data.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE mobile=? AND password=?", (mobile, password))
            user = cursor.fetchone()
            conn.close()

            if user:
                session['user_id'] = user[0]
                session['user_name'] = user[1]
                return redirect(url_for('document_upload'))  # Change to your actual upload page route
            else:
                error = 'Invalid credentials'

        elif 'register' in request.form:
            # Register form submitted
            name = request.form['reg_name']
            mobile = request.form['reg_mobile']
            password = request.form['reg_password']

            conn = sqlite3.connect('user_data.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, mobile, password) VALUES (?, ?, ?)",
                           (name, mobile, password))
            conn.commit()
            conn.close()
            return redirect(url_for('document_upload', _anchor='login'))  # Redirect back to login section

    return render_template('auth.html', error=error)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

# Aadhaar Match Check
def check_aadhaar_match(aadhaar_number):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM aadhaar_data WHERE REPLACE(aadhaar_number, ' ', '') = ?", (aadhaar_number.replace(" ", ""),))
    result = cursor.fetchone()
    conn.close()
    return "✅  Aadhaar Card Match Found" if result else "❌ Aadhaar Card  Match"

# Ration Card Match Check
def check_rationcard_match(card_number):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM ration_data WHERE REPLACE(card_number, ' ', '') = ?", (card_number.replace(" ", ""),))
    result = cursor.fetchone()
    conn.close()
    return "✅  Ration Card Match Found" if result else "❌  Ration Card Match"
   



@app.route("/document_upload", methods=["GET", "POST"])
def document_upload():
    aadhaar_data = {}
    smart_data = {}
    aadhaar_match_result = ""
    ration_match_result = ""
    cross_match = ""

    if request.method == "POST":
        aadhaar_file = request.files.get("aadhaarFile")
        smart_file = request.files.get("smartFile")

        # Aadhaar File Processing
        if aadhaar_file and allowed_file(aadhaar_file.filename):
            aadhaar_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(aadhaar_file.filename))
            aadhaar_file.save(aadhaar_path)

            qr_text = extract_qr_data(aadhaar_path)
            aadhaar_data = parse_aadhaar_data(qr_text)

            # Check Aadhaar match in DB
            aadhaar_number = aadhaar_data.get('aadhaar_number', '')
            if aadhaar_number:
                aadhaar_match_result = check_aadhaar_match(aadhaar_number)

        # Smart Card File Processing
        if smart_file and allowed_file(smart_file.filename):
            smart_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(smart_file.filename))
            smart_file.save(smart_path)

            smart_text = extract_qr_data(smart_path)
            smart_data = parse_smartcard_data(smart_text)

            # Check Smart Card match in DB
            card_number = smart_data.get('card_number', '')
            if card_number:
                ration_match_result = check_rationcard_match(card_number)

        # Save to Session
        if aadhaar_data and smart_data:
            save_session_data(session, aadhaar_data, smart_data)

        # Cross Verification
        if aadhaar_data and smart_data:
            aadhaar_name = aadhaar_data.get('user_name', '').lower()
            smart_head = smart_data.get('head_of_family', '').lower()

            aadhaar_address = aadhaar_data.get('address', '').lower()
            smart_address = smart_data.get('address', '').lower()

            address_match = any(word in smart_address for word in aadhaar_address.split() if len(word) > 3)
            name_match = any(part in smart_head for part in aadhaar_name.split() if len(part) > 3)
            dob_match = bool(aadhaar_data.get('dob', '').strip())

            if name_match and (address_match or dob_match):
                cross_match = "✅ Aadhaar & Smart Card Info Match"
            else:
                issues = []
                if not name_match: issues.append("name")
                if not address_match: issues.append("address")
                if not dob_match: issues.append("DOB")
                cross_match = f"❌ Mismatch in: {', '.join(issues)}"

    return render_template("document_upload.html",
                           aadhaar=aadhaar_data,
                           smart=smart_data,
                           aadhaar_match=aadhaar_match_result,
                           ration_match=ration_match_result,
                           cross_match=cross_match)



@app.route("/user_form", methods=["GET", "POST"])
def user_form():
    if request.method == "POST":
        # Get form data from the form submission
        occupation = request.form.get('occupation', '')
        annual_income_range = request.form.get('annual_income_range', '')

        # Save to session
        session['occupation'] = occupation
        session['annual_income_range'] = annual_income_range
        session.modified = True

        print(f"DEBUG – querying for occupation={occupation}, annual_income={annual_income_range}")

        # Redirect to the next step (update with your actual route)
        return redirect(url_for("bank_data"))

    # For GET request, render the form with values from session
    return render_template("loan_form.html",
        aadhaar_number=session.get('aadhaar_number', ''),
        card_number=session.get('card_number', ''),
        user_name=session.get('user_name', ''),
        mobile=session.get('mobile', ''),
        card_type=session.get('card_type', ''),
        address=session.get('address', ''),
        occupation=session.get('occupation', ''),
        annual_income_range=session.get('annual_income_range', '')
    )




@app.route('/bank_data', methods=['GET', 'POST'])
def bank_data():
    aadhaar_number = session.get('aadhaar_number')
    card_number   = session.get('card_number')
    occupation   = session.get('occupation')
    annual_income_range = session.get('annual_income_range')

    # Quick sanity‑check
    if not aadhaar_number or not card_number:
        return render_template(
            'error.html',
            error="❌ Aadhaar number and Card number not found in session.",
            back_url=url_for('user_form')
        )

    # Remove any spaces before querying
    aadhaar_number = aadhaar_number.replace(" ", "")
    card_number    = card_number.replace(" ", "")

    print(f"DEBUG – querying for Aadhaar={aadhaar_number}, Card={card_number}")

    # ── 2.  Fetch from DB ─────────────────────────────────────────────
    conn   = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, aadhaar_number, Card_Number, Address, mobile_number,
               account_number, bank_name, branch, IFSC_Code,
               Account_type, subsidy_Applied
        FROM   Bank_data
        WHERE  REPLACE(aadhaar_number,' ','') = ?
        AND    REPLACE(Card_Number,' ','')    = ?
    """, (aadhaar_number, card_number))
    loan_records = cursor.fetchall()

    # fallback: search only by Aadhaar
    if not loan_records:
        cursor.execute("""
            SELECT name, aadhaar_number, Card_Number, Address, mobile_number,
                   account_number, bank_name, branch, IFSC_Code,
                   Account_type, subsidy_Applied
            FROM   Bank_data
            WHERE  REPLACE(aadhaar_number,' ','') = ?
        """, (aadhaar_number,))
        loan_records = cursor.fetchall()

    conn.close()

    # ── 3.  Handle “no record” case ───────────────────────────────────
    if not loan_records:
        return render_template(
            'fetch_data.html',
            loan_records=[],
            error="❌ No matching loan record found for the given Aadhaar and Card numbers."
        )

    # ── 4.  SAVE RESULTS TO SESSION  ──────────────────────────────────
    #
    # Turn the first row into a dict so we can stash it easily.
    # (If you may return multiple rows, iterate and store a list.)
    cols = [
        "name","aadhaar_number","card_number","address","mobile_number",
        "account_number","bank_name","branch","ifsc_code",
        "account_type","subsidy_applied"
    ]
    record_dict = dict(zip(cols, loan_records[0]))

    # Persist each field in session
    for k, v in record_dict.items():
        session[k] = v

    session.modified = True      # force‑save



    # ── 5.  Show the page  ────────────────────────────────────────────

    return render_template(
        'fetch_data.html',
        loan_records=loan_records,
        occupation=occupation,
        annual_income_range=annual_income_range,
        error=None
    )




#--------------------------------------------------------------------------------------------------------#
@app.route('/select_scheme', methods=['GET', 'POST'])
def select_scheme():
    if request.method == 'POST':
        selected_scheme = request.form.get('scheme')  # 'agriculture', 'gold', etc.
        session['selected_scheme'] = selected_scheme

        if selected_scheme == 'Agriculture':
            return redirect(url_for('loan_waiver_form'))
        elif selected_scheme == 'Gold':
            return redirect(url_for('gold_loan_form'))
        elif selected_scheme == 'shg':
            return redirect(url_for('shg_form'))
    
    # For GET requests, render the scheme selection template
    return render_template('scheme_selection.html', selected_scheme=session.get('selected_scheme'))
  #---------------------------------------------------------------------------------------------#  
  # Load models and feature names once
def load_model_and_features(model_path, feature_path):
    """Helper function to load model and feature names"""
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
            
        # Load model
        model = joblib.load(model_path)
        
        # Load feature names
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)
            
        return model, features
    except Exception as e:
        print(f"Error loading model or features: {str(e)}")
        raise

# Route 1: Agriculture Loan Waiver Form
@app.route('/loan_waiver_form', methods=['GET', 'POST'])
def loan_waiver_form():
    annual_income = session.get
    if request.method == 'POST':
        try:
            # Debug the form data
            print("\n===== FORM DATA RECEIVED =====")
            for key, value in request.form.items():
                print(f"{key}: {value}")
            print("=====\n")
            
            # Collect data from form with default values for required fields
            loan_data = {
                'Loan_Amount': request.form.get('loan_amount', '<2Lakh'),  # Default value
                'Land_Size': request.form.get('land_size', '1-2'),  # Default value
                'Repayment_Status': request.form.get('repayment_status', 'Fully Paid'),  # Default value
                'Loan_Tenure': request.form.get('loan_tenure', 'less than 3 yrs'),  # Default value
                'Collateral_Type': request.form.get('collateral_type', 'Land'),  # Default value
                'Collateral_Value': request.form.get('collateral_value', '>1Lakh'),  # Default value
                'Subsidy': request.form.get('subsidy_type', 'No'),  # Default value
                'Loan_Purpose': request.form.get('loan_purpose', 'Agriculture'),  # Default value
                'Card_Type': request.form.get('ration_card_type', 'NPHH'),  # Default value
                'Income': request.form.get('annual_income', 'less than 70k'),  # Default value
                'Previous_Loan_History': request.form.get('previous_loan_history', 'No Loan'),  # Default value
                'Interest_Rate': 'less than 4'  # Default value
            }
            
            # Ensure no empty or None values
            for key in loan_data:
                if not loan_data[key]:
                    # Provide sensible defaults for empty values
                    if key == 'Loan_Amount':
                        loan_data[key] = '<2Lakh'
                    elif key == 'Land_Size':
                        loan_data[key] = '1-2'
                    elif key == 'Repayment_Status':
                        loan_data[key] = 'Fully Paid'
                    elif key == 'Loan_Tenure':
                        loan_data[key] = 'less than 3 yrs'
                    elif key == 'Collateral_Type':
                        loan_data[key] = 'Land'
                    elif key == 'Collateral_Value':
                        loan_data[key] = '>1Lakh'
                    elif key == 'Subsidy':
                        loan_data[key] = 'No'
                    elif key == 'Loan_Purpose':
                        loan_data[key] = 'Agriculture'
                    elif key == 'Card_Type':
                        loan_data[key] = 'NPHH'
                    elif key == 'Income':
                        loan_data[key] = 'less than 70k'
                    elif key == 'Previous_Loan_History':
                        loan_data[key] = 'No Loan'
                    elif key == 'Interest_Rate':
                        loan_data[key] = 'less than 4'
            
            # Map form values to model expected values if needed
            # For example, if the form has "1-2 acres" but model expects "1-2"
            land_size_mapping = {
                'less than 1 acre': '1-2',
                '1-2 acres': '1-2',
                '2-5 acres': '2-4',
                'more than 5 acres': '5-6'
            }
            
            if loan_data['Land_Size'] in land_size_mapping:
                loan_data['Land_Size'] = land_size_mapping[loan_data['Land_Size']]
                
            # Map repayment status
            repayment_mapping = {
                'Regular': 'Fully Paid',
                'Defaulted': 'Default',
                'Partially Paid': 'Partially Paid'
            }
            
            if loan_data['Repayment_Status'] in repayment_mapping:
                loan_data['Repayment_Status'] = repayment_mapping[loan_data['Repayment_Status']]
                
            # Map loan tenure
            tenure_mapping = {
                '1 year': 'less than 3 yrs',
                '2 years': 'less than 3 yrs',
                '3 years': 'less than 3 yrs',
                'more than 3 years': 'above 3 yrs'
            }
            
            if loan_data['Loan_Tenure'] in tenure_mapping:
                loan_data['Loan_Tenure'] = tenure_mapping[loan_data['Loan_Tenure']]
                
            # Map collateral value
            collateral_mapping = {
                'less than loan': '<50K',
                'equal to loan': '60K-1Lakh',
                'more than loan': '>1Lakh'
            }
            
            if loan_data['Collateral_Value'] in collateral_mapping:
                loan_data['Collateral_Value'] = collateral_mapping[loan_data['Collateral_Value']]
                
            # Map card type
            card_mapping = {
                'APL': 'NPHH',
                'BPL': 'PHH',
                'Antyodaya': 'PHH-AYY'
            }
            
            if loan_data['Card_Type'] in card_mapping:
                loan_data['Card_Type'] = card_mapping[loan_data['Card_Type']]
                
            # Map income
            income_mapping = {
                'below 80k': 'less than 70k',
                '80k-1.5Lakh': '80k-1Lakh',
                'above 1.5Lakh': 'above 1.5Lakh'
            }
            
            if loan_data['Income'] in income_mapping:
                loan_data['Income'] = income_mapping[loan_data['Income']]
                
            # Map previous loan history
            history_mapping = {
                'No previous loans': 'No Loan',
                'All loans repaid': '1-2 Loans',
                'Some defaults': '1-2 Loans',
                'Multiple defaults': '>2 Loans'
            }
            
            if loan_data['Previous_Loan_History'] in history_mapping:
                loan_data['Previous_Loan_History'] = history_mapping[loan_data['Previous_Loan_History']]
            
            # Debugging: Print the mapped data
            print("\n===== MAPPED DATA =====")
            for key, value in loan_data.items():
                print(f"{key}: {value}")
                # Store in session for later use
                session[key] = value
            print("=====\n")
            
            # Force session to save
            session.modified = True
            
            try:
                # Create model directory if it doesn't exist
                os.makedirs('model', exist_ok=True)
                
                # Check if model files exist, if not, copy from models directory
                if not os.path.exists('model/loan_eligibility_model.pkl') and os.path.exists('models/loan_eligibility_model.pkl'):
                    import shutil
                    shutil.copy('models/loan_eligibility_model.pkl', 'model/loan_eligibility_model.pkl')
                    
                if not os.path.exists('model/feature_names.pkl') and os.path.exists('models/feature_names.pkl'):
                    import shutil
                    shutil.copy('models/feature_names.pkl', 'model/feature_names.pkl')
                
                # Load the model and feature names
                model_path = 'model/loan_eligibility_model.pkl'
                feature_path = 'model/feature_names.pkl'
                
                print(f"Loading model from {model_path} and features from {feature_path}")
                
                # Check if model files exist
                if not os.path.exists(model_path) or not os.path.exists(feature_path):
                    raise FileNotFoundError(f"Model files not found: {model_path} or {feature_path}")
                
                model, features = load_model_and_features(model_path, feature_path)
                
                # Debug features
                print(f"Features expected: {features}")
                
                # Use pandas to handle the one-hot encoding properly
                import pandas as pd
                
                # Create a DataFrame with the input data
                input_df = pd.DataFrame([loan_data])
                
                # Print the DataFrame for debugging
                print("Input DataFrame before encoding:")
                print(input_df)
                
                # Get the dummy variables (one-hot encoding)
                input_encoded = pd.get_dummies(input_df)
                
                # Print the encoded DataFrame for debugging
                print("Input DataFrame after encoding:")
                print(input_encoded)
                
                # Align with the training data columns
                missing_cols = set(features) - set(input_encoded.columns)
                for col in missing_cols:
                    input_encoded[col] = 0
                
                # Make sure we only use the columns that the model expects
                input_encoded = input_encoded[features]
                
                print("[DEBUG] Encoded input data shape:", input_encoded.shape)
                print("[DEBUG] Encoded input data columns:", input_encoded.columns.tolist())
                
                # Make prediction
                prediction = model.predict(input_encoded)[0]
                probability = model.predict_proba(input_encoded)[0][1]
                
                # Generate explanation based on the loan_eligibility_model.py logic
                if prediction:
                    explanation = f"You are eligible for the loan waiver with {probability:.1%} confidence.\n"
                    
                    # Add specific reasons
                    if loan_data.get('Subsidy') == 'Yes':
                        explanation += "- You are already receiving a subsidy, which qualifies you for the waiver.\n"
                        
                    if loan_data.get('Income') == 'less than 70k':
                        explanation += "- Your income level (below ₹70,000) qualifies you for assistance.\n"
                        
                    if loan_data.get('Card_Type') in ['PHH', 'PHH-AYY']:
                        explanation += "- Your ration card type (Priority Household) makes you eligible.\n"
                        
                    if loan_data.get('Loan_Tenure') == 'less than 3 yrs':
                        explanation += "- Your loan tenure (less than 3 years) meets the eligibility criteria.\n"
                        
                    if loan_data.get('Collateral_Type') == 'Gold' and loan_data.get('Repayment_Status') == 'Default':
                        explanation += "- Your gold collateral with default status qualifies for the waiver program.\n"
                else:
                    explanation = f"You are not eligible for the loan waiver with {(1-probability):.1%} confidence.\n"
                    
                    # Add specific reasons for rejection
                    if loan_data.get('Income') == 'above 1.5Lakh':
                        explanation += "- Your income exceeds the maximum threshold (₹1.5 Lakh) for this waiver.\n"
                        
                    if loan_data.get('Loan_Amount') == '<2Lakh':
                        explanation += "- Your loan amount exceeds the maximum covered under this scheme.\n"
                        
                    if loan_data.get('Loan_Tenure') == 'above 3 yrs':
                        explanation += "- Your loan tenure (above 3 years) exceeds the eligible period.\n"
                        
                    if loan_data.get('Repayment_Status') == 'Fully Paid':
                        explanation += "- Your loan is already fully paid, so no waiver is applicable.\n"
                        
                    if loan_data.get('Land_Size') == '5-6' and loan_data.get('Income') == 'above 1.5Lakh':
                        explanation += "- Your land size and income level combined make you ineligible.\n"
                
                # Store results in session
                session['eligibility'] = "Eligible" if prediction else "Not Eligible"
                session['eligibility_probability'] = f"{probability:.1%}"
                session['eligibility_explanation'] = explanation
                
                # Redirect to results page
                return redirect(url_for('eligibility_results'))
                
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Even if prediction fails, redirect to results with error message
                session['eligibility'] = "Error"
                session['eligibility_probability'] = "N/A"
                session['eligibility_explanation'] = f"An error occurred: {str(e)}"
                return redirect(url_for('eligibility_results'))
                
        except Exception as e:
            print(f"Unexpected error in loan_waiver_form: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Even if there's an error, redirect to results with error message
            session['eligibility'] = "Error"
            session['eligibility_probability'] = "N/A"
            session['eligibility_explanation'] = f"An unexpected error occurred: {str(e)}"
            return redirect(url_for('eligibility_results'))
    
    # For GET requests, render the form template
    return render_template('agri_loan.html')  # Route 2: Gold Loan Waiver Form
@app.route('/gold_loan_form', methods=['GET', 'POST'])
def gold_loan_form():
    # Add error handling at the beginning
    error = None
    
    if request.method == 'POST':
        try:
            # Debug the form data
            print("\n===== FORM DATA RECEIVED =====")
            for key, value in request.form.items():
                print(f"{key}: {value}")
            print("=====\n")
            
            # Collect data with defaults for missing fields
            gold_data = {
                'Loan_Amount': request.form.get('loan_amount', 'below 80k'),
                'Gold_Type': request.form.get('gold_type', 'Ornament'),
                'Gold_Weight': request.form.get('gold_weight', 'less than 20g'),
                'Gold_Purity': request.form.get('gold_purity', '22K'),
                'Years_of_Repayment': request.form.get('repayment_years', '1'),
                'Interest_Rate': request.form.get('interest_rate', 'less than 7'),
                'Subsidy': request.form.get('subsidy_type', 'No'),
                'Card_Type': request.form.get('ration_card_type', 'PHH'),
                'Annual_Income': request.form.get('annual_income', 'below 80k'),
                'Provider_Bank_Name': request.form.get('provider_bank', 'Canara'),
                'Workers': 'Farmer',  # Default value
                'Purpose_of_loan': 'Medical',  # Default value
                'Pension': 'No',  # Default value
                'Already_Loan_Waived': 'No'  # Default value
            }
            
            # Ensure no empty or None values
            for key in gold_data:
                if not gold_data[key]:
                    # Provide sensible defaults for empty values
                    if key == 'Loan_Amount':
                        gold_data[key] = 'below 80k'
                    elif key == 'Gold_Type':
                        gold_data[key] = 'Ornament'
                    elif key == 'Gold_Weight':
                        gold_data[key] = 'less than 20g'
                    elif key == 'Gold_Purity':
                        gold_data[key] = '22K'
                    elif key == 'Years_of_Repayment':
                        gold_data[key] = '1'
                    elif key == 'Interest_Rate':
                        gold_data[key] = 'less than 7'
                    elif key == 'Subsidy':
                        gold_data[key] = 'No'
                    elif key == 'Card_Type':
                        gold_data[key] = 'PHH'
                    elif key == 'Annual_Income':
                        gold_data[key] = 'below 80k'
                    elif key == 'Provider_Bank_Name':
                        gold_data[key] = 'Canara'
                    elif key == 'Workers':
                        gold_data[key] = 'Farmer'
                    elif key == 'Purpose_of_loan':
                        gold_data[key] = 'Medical'
                    elif key == 'Pension':
                        gold_data[key] = 'No'
                    elif key == 'Already_Loan_Waived':
                        gold_data[key] = 'No'

            print("\n===== User Input Data (Gold) =====")
            for key, value in gold_data.items():
                print(f"{key}: {value}")
                # Store in session for later use
                session[key] = value
            print("=====\n")
            
            # Force session to save
            session.modified = True
            
            try:
                # Create model directory if it doesn't exist
                os.makedirs('model', exist_ok=True)
                
                # Check if model files exist, if not, copy from models directory
                if not os.path.exists('model/gold_eligibility_model.pkl') and os.path.exists('models/gold_eligibility_model.pkl'):
                    import shutil
                    shutil.copy('models/gold_eligibility_model.pkl', 'model/gold_eligibility_model.pkl')
                    
                if not os.path.exists('model/feature_gold_names.pkl') and os.path.exists('models/feature_gold_names.pkl'):
                    import shutil
                    shutil.copy('models/feature_gold_names.pkl', 'model/feature_gold_names.pkl')
                
                # Prepare input data
                model_path = 'model/gold_eligibility_model.pkl'
                feature_path = 'model/feature_gold_names.pkl'
                
                print(f"Loading model from {model_path} and features from {feature_path}")
                
                # Check if model files exist
                if not os.path.exists(model_path) or not os.path.exists(feature_path):
                    raise FileNotFoundError(f"Model files not found: {model_path} or {feature_path}")
                
                model, features = load_model_and_features(model_path, feature_path)
                
                # Debug features
                print(f"Features expected: {features}")
                
                # Use pandas to handle the one-hot encoding properly
                import pandas as pd
                
                # Create a DataFrame with the input data
                input_df = pd.DataFrame([gold_data])
                
                # Print the DataFrame for debugging
                print("Input DataFrame before encoding:")
                print(input_df)
                
                # Get the dummy variables (one-hot encoding)
                input_encoded = pd.get_dummies(input_df)
                
                # Print the encoded DataFrame for debugging
                print("Input DataFrame after encoding:")
                print(input_encoded)
                
                # Align with the training data columns
                missing_cols = set(features) - set(input_encoded.columns)
                for col in missing_cols:
                    input_encoded[col] = 0
                
                # Make sure we only use the columns that the model expects
                input_encoded = input_encoded[features]
                
                print("[DEBUG] Encoded input data shape:", input_encoded.shape)
                print("[DEBUG] Encoded input data columns:", input_encoded.columns.tolist())

                # Make prediction
                prediction = model.predict(input_encoded)[0]
                probability = model.predict_proba(input_encoded)[0][1]
                
                # Generate explanation based on gold_loan_prediction_model.py logic
                if prediction:
                    explanation = f"You are eligible for the gold loan waiver with {probability:.1%} confidence.\n"
                    
                    # Add specific reasons
                    if gold_data.get('Gold_Weight') == 'less than 20g' or gold_data.get('Gold_Weight') == '20g to 40g':
                        explanation += "- Your gold weight qualifies you for the waiver.\n"
                        
                    if gold_data.get('Gold_Purity') in ['22K', '24K', '18K']:
                        explanation += "- Your gold purity meets the eligibility criteria.\n"
                        
                    if gold_data.get('Workers') in ['Farmer', 'Housewife', 'Small Business', 'daily wages']:
                        explanation += "- Your occupation qualifies you for the waiver.\n"
                        
                    if gold_data.get('Pension') == 'Yes':
                        explanation += "- Your pension status meets the eligibility criteria.\n"
                        
                    if gold_data.get('Years_of_Repayment') in ['1', '2']:
                        explanation += "- Your years of repayment meets the eligibility criteria.\n"
                        
                    if gold_data.get('Annual_Income') == 'below 80k':
                        explanation += "- Your annual income meets the eligibility criteria.\n"
                        
                    if gold_data.get('Provider_Bank_Name') not in ['SBI', 'Axis bank']:
                        explanation += "- Your provider bank meets the eligibility criteria.\n"
                else:
                    explanation = f"You are not eligible for the gold loan waiver with {(1-probability):.1%} confidence.\n"
                    
                    # Add specific reasons for rejection
                    if gold_data.get('Subsidy') == 'Yes':
                        explanation += "- You are already receiving a subsidy, which makes you ineligible.\n"
                        
                    if gold_data.get('Already_Loan_Waived') == 'Yes':
                        explanation += "- You have already availed a loan waiver, which makes you ineligible.\n"
                        
                    if gold_data.get('Gold_Weight') == 'above 40g':
                        explanation += "- Your gold weight exceeds the maximum covered under this scheme.\n"
                        
                    if gold_data.get('Annual_Income') == 'above 80k':
                        explanation += "- Your annual income exceeds the maximum threshold for this waiver.\n"
                        
                    if gold_data.get('Provider_Bank_Name') in ['SBI', 'Axis bank']:
                        explanation += "- Your provider bank is not eligible for this waiver scheme.\n"
                        
                    if gold_data.get('Workers') in ['Government staff', 'Banker']:
                        explanation += "- Your occupation makes you ineligible for this waiver.\n"
                
                # Store results in session
                session['eligibility'] = "Eligible" if prediction else "Not Eligible"
                session['eligibility_probability'] = f"{probability:.1%}"
                session['eligibility_explanation'] = explanation
                
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Even if prediction fails, set error message
                session['eligibility'] = "Error"
                session['eligibility_probability'] = "N/A"
                session['eligibility_explanation'] = f"An error occurred: {str(e)}"
            
            # Always redirect to results page after POST
            return redirect(url_for('eligibility_results'))
                
        except Exception as e:
            # Catch any other exceptions in the POST handling
            print(f"Unexpected error in gold_loan_form POST: {str(e)}")
            import traceback
            traceback.print_exc()
            error = f"An unexpected error occurred: {str(e)}"
    
    # For GET requests or if there was an error, render the form template
    return render_template('gold_loan.html', error=error)

@app.route('/shg_form', methods=['GET', 'POST'])
def shg_form():
    error = None

    if request.method == 'POST':
        try:
            # Debug the form data
            print("\n===== SHG FORM DATA RECEIVED =====")
            for key, value in request.form.items():
                print(f"{key}: {value}")
            print("=====\n")

            # Form processing - collect all form data
            shg_data = {
                'formation_year': request.form.get('formation_year'),
                'member_count': request.form.get('member_count'),
                'loan_amount': request.form.get('loan_amount'),
                'loan_provider_bank': request.form.get('loan_provider_bank'),
                'annual_income': request.form.get('annual_income', 'less than ₹90K'),
                'loan_usage': request.form.get('loan_usage')
            }

            # Save to session
            for key, value in request.form.items():
                session[key] = value
            session.modified = True

            # Try model prediction
            try:
                from shg_loan import predict_eligibility
                eligibility, probability, explanation = predict_eligibility(shg_data)

                # Store results in session
                session['eligibility'] = eligibility
                session['eligibility_probability'] = f"{probability:.1%}"
                session['eligibility_explanation'] = explanation
                
                print(f"Prediction result: {eligibility}, {probability:.1%}")
                print(f"Explanation: {explanation}")

            except Exception as e:
                print(f"Model Prediction Failed: {str(e)}")
                import traceback
                traceback.print_exc()

                # Set error message
                session['eligibility'] = "Error"
                session['eligibility_probability'] = "N/A"
                session['eligibility_explanation'] = f"An error occurred: {str(e)}"

            return redirect(url_for('eligibility_results'))

        except Exception as e:
            print(f"Error in shg_form: {str(e)}")
            import traceback
            traceback.print_exc()

            session['eligibility'] = "Error"
            session['eligibility_probability'] = "N/A"
            session['eligibility_explanation'] = f"An error occurred: {str(e)}"
            return redirect(url_for('eligibility_results'))

    return render_template('shg.html', error=error)

@app.route('/eligibility_results', methods=['GET', 'POST'])
def eligibility_results():
    # Initialize variables with default values from session
    eligibility = session.get('eligibility', 'Unknown')
    probability = session.get('eligibility_probability', '0%')
    explanation = session.get('eligibility_explanation', 'No explanation available.')
    
    # Get user data from session (for display)
    user_name = session.get('user_name', 'Unknown')
    aadhaar_number = session.get('aadhaar_number', 'Unknown')
    card_number = session.get('card_number', 'Unknown')
    card_type = session.get('card_type', 'Unknown')
    mobile = session.get('mobile', 'Unknown')
    address = session.get('address', 'Unknown')
    occupation = session.get('occupation', 'Unknown')
    annual_income = session.get('annual_income', 'Unknown')

    if request.method == 'POST':
        # Debug: Print session data to make sure it's correct
        print("\n===== ELIGIBILITY RESULTS =====")
        print(f"Eligibility: {eligibility}")
        print(f"Probability: {probability}")
        print(f"Explanation: {explanation}")
        print("=====\n")

        print("\n===== USER DATA FORM SUBMISSION =====")
        for key, value in request.form.items():
            print(f"{key}: {value}")
            # Save every field to session
            session[key] = value or ""    # blank string instead of None
            # Force session to save
            session.modified = True
        
    return render_template(
        'eligibility_results.html',
        eligibility=eligibility,
        probability=probability,
        explanation=explanation,
        user_name=user_name,
        aadhaar_number=aadhaar_number,
        card_number=card_number,
        card_type=card_type,
        mobile=mobile,
        address=address,
        occupation=occupation,
        annual_income=annual_income,
        session_data=session
    )


# ---------- PREVIEW -------------------------------------------------
@app.route('/eligibility_results_preview')
def eligibility_results_preview():
    """
    Same data as /eligibility_results, but rendered in a printable / downloadable
    layout with a Download button.
    """
    # Pull all keys you already saved in session — add more as needed
    context = {
        "eligibility":             session.get('eligibility', 'Unknown'),
        "probability":             session.get('eligibility_probability', '0%'),
        "explanation":             session.get('eligibility_explanation', 'No explanation'),
        "user_name":               session.get('user_name', 'Unknown'),
        "aadhaar_number":          session.get('aadhaar_number', 'Unknown'),
        "card_number":             session.get('card_number', 'Unknown'),
        "card_type":               session.get('card_type', 'Unknown'),
        "mobile":                  session.get('mobile', 'Unknown'),
        "address":                 session.get('address', 'Unknown'),
        "occupation":              session.get('occupation', 'Unknown'),
        "annual_income":           session.get('annual_income', 'Unknown'),
    }
    
    
    
    return render_template("result_preview.html", **context)


# ---------- DOWNLOAD ------------------------------------------------
@app.route('/eligibility_results_download')
def eligibility_results_download():
    """
    Render the same preview template to HTML → convert to PDF → stream it.
    """
    context  = {
        "eligibility":             session.get('eligibility', 'Unknown'),
        "probability":             session.get('eligibility_probability', '0%'),
        "explanation":             session.get('eligibility_explanation', 'No explanation'),
        "user_name":               session.get('user_name', 'Unknown'),
        "aadhaar_number":          session.get('aadhaar_number', 'Unknown'),
        "card_number":             session.get('card_number', 'Unknown'),
        "card_type":               session.get('card_type', 'Unknown'),
        "mobile":                  session.get('mobile', 'Unknown'),
        "address":                 session.get('address', 'Unknown'),
        "occupation":              session.get('occupation', 'Unknown'),
        "annual_income":           session.get('annual_income', 'Unknown'),
    }

    # Render the HTML with Jinja
    html = render_template("result_preview.html", **context)

    # Convert to PDF (wkhtmltopdf must be installed)
    pdf_bytes = pdfkit.from_string(html, False, configuration=config)

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="eligibility_summary.pdf"
    )

# -------------------- STORE TO DASHBOARD DB --------------------
def add_user(username, account_creation, aadhar_image_path, match_result):
    con = sqlite3.connect('dashboard.db')
    cursor = con.cursor()
    cursor.execute("INSERT INTO users (username, account_creation, aadhar_image_path, match_result) VALUES (?, ?, ?, ?)",
                   (username, account_creation, aadhar_image_path, match_result))
    con.commit()
    con.close()


# -------------------- BANKER LOGIN --------------------
@app.route('/banker', methods=['GET', 'POST'])
def banker():
    if request.method == 'POST':
        name = request.form['name']
        bank_id = request.form['bank_id']
        password = request.form['password']
        designation = request.form['designation']

        conn = sqlite3.connect('user_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bankers WHERE name=? AND bank_id=? AND password=? AND designation=?",
                       (name, bank_id, password, designation))
        banker = cursor.fetchone()
        conn.close()

        if banker:
            session['banker_name'] = name
            session['banker_id'] = bank_id
            return redirect(url_for('dashboard'))
        else:
            return render_template('banker.html', error="Invalid credentials")
    return render_template('banker.html')


@app.route('/dashboard')
def dashboard():
    con = sqlite3.connect('dashboard.db')
    cursor = con.cursor()
    cursor.execute("SELECT * FROM users")
    user_list = cursor.fetchall()
    con.close()

    return render_template('dashboard.html', user_list=user_list, banker_name=session.get('banker_name'),
                           banker_id=session.get('banker_id'))


@app.route('/view_user/<int:user_id>', methods=['GET'])
def view_user(user_id):
    con = sqlite3.connect('dashboard.db')
    cursor = con.cursor()
    cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()
    con.close()
    return render_template('view_user.html', user=user)


# -------------------- LOGOUT --------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# -------------------- RUN APP --------------------
if __name__ == '__main__':
    app.run(debug=True, port=5001)

# Add this function to your app.py to handle string to numeric conversion safely
def safe_convert_to_numeric(value):
    """Safely convert string values to numeric for model input"""
    if isinstance(value, (int, float)):
        return value
        
    # Handle loan amount strings
    if isinstance(value, str):
        if 'below' in value.lower() or 'less than' in value.lower():
            return 1  # Lowest category
        elif 'lakh' in value.lower() and ('1' in value or 'one' in value.lower()):
            return 2  # Middle category
        elif 'above' in value.lower() or 'more than' in value.lower():
            return 3  # Highest category
        elif value.isdigit():
            return int(value)
            
    # Default value if conversion fails
    return 1  # Default to lowest category

# Example of using this in prediction code
def predict_shg_eligibility(shg_data):
    try:
        # Convert string values to appropriate numeric values
        processed_data = {
            'member_count': safe_convert_to_numeric(shg_data.get('member_count', 1)),
            'loan_amount': safe_convert_to_numeric(shg_data.get('loan_amount', 'below ₹1 Lakh')),
            'loan_purpose': shg_data.get('loan_purpose', 'Livelihood'),
            'repayment_years': safe_convert_to_numeric(shg_data.get('repayment_years', 1)),
            'members_count': safe_convert_to_numeric(shg_data.get('members_count', 5)),
            'savings_amount': safe_convert_to_numeric(shg_data.get('savings_amount', 10000)),
            'bank_name': shg_data.get('bank_name', 'Grama Bank'),
            'subsidy_received': shg_data.get('subsidy_received', 'No')
        }
        
        # Create DataFrame with processed data
        import pandas as pd
        df = pd.DataFrame([processed_data])
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df)
        
        # Load model and make prediction
        import joblib
        model = joblib.load('shg_loan_model.pkl')
        
        # Ensure all expected columns are present
        # This would need to be adjusted based on your actual model's expected features
        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Select only the columns the model expects
        df_encoded = df_encoded[expected_columns]
        
        # Make prediction
        prediction = model.predict(df_encoded)[0]
        probability = model.predict_proba(df_encoded)[0][1]
        
        return prediction, probability
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0

























