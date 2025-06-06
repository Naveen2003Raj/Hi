import sqlite3
import csv
import os

csv_file = 'final_bank_data.csv'  # Change this to your CSV file path (put your CSV in the same folder or specify full path)
db_file = 'user_data.db'

# Check if DB exists and data is already inserted
def is_data_already_inserted():
    if not os.path.exists(db_file):
        return False
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM Bank_data")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except sqlite3.OperationalError:
        conn.close()
        return False

# Main Upload Function
def upload_csv_once():
    if is_data_already_inserted():
        print("✅ CSV already uploaded. No action needed.")
        return

    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS Bank_data")

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Bank_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            aadhaar_number TEXT,
            card_number TEXT,
            Address TEXT,
            Mobile_Number TEXT,
            account_number TEXT,
            bank_name TEXT,
            subsidy_Applied TEXT,
            IFSC_Code TEXT,
            Branch TEXT,
            Account_Type TEXT
        )
    ''')
    
    # Print first row of CSV to debug
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        first_row = next(reader) if reader else []
        print("CSV Headers:", headers)
        print("First row:", first_row)

    # Read CSV and insert data
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Print row for debugging
            print(f"Processing row: {dict(row)}")
            
            cursor.execute('''
                INSERT INTO Bank_data (
                    Name, 
                    card_number, 
                    aadhaar_number,
                    Address,
                    Mobile_Number,
                    account_number,
                    bank_name,
                    subsidy_Applied,
                    IFSC_Code,
                    Branch,
                    Account_Type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['Name'],  # Using family_head as name
                row['card_number'],  # Using card_number from CSV
                row['aadhaar_number'],
                row['Address'],
                row['Mobile_Number'],
                row['account_number'],
                row['bank_name'],
                row['subsidy_applied'],
                row['IFSC_Code'],
                row['Branch'],
                row['Account_Type']
            ))

    conn.commit()
    conn.close()
    print("✅ CSV uploaded successfully into user_data.db")

# Run only once
upload_csv_once()
