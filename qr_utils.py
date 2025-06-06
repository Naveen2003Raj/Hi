import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import re

def extract_qr_data(image_path):
    """Extracts QR code text from the given image."""
    img = cv2.imread(image_path)
    decoded = pyzbar.decode(img)
    if decoded:
        return decoded[0].data.decode("utf-8")
    return ""


def parse_aadhaar_data(data):
    """Parses Aadhaar QR text data into structured dictionary."""
    result = {
        "user_name": "",
        "aadhaar_number": "",
        "dob": "",
        "gender": "",
        "mobile": "",
        "address": ""
    }

    dob_keywords = ['dob', 'dateofbirth', 'birth', 'yob', 'yearofbirth']

    for field in data.split('\n'):
        field_lower = field.lower()

        # Name
        if any(keyword in field_lower for keyword in ['name', 'nam']):
            result['user_name'] = field.split(':')[-1].strip()

        # Aadhaar Number
        elif any(keyword in field_lower for keyword in ['aadhaar', 'uid', 'vid']):
            digits = ''.join(c for c in field if c.isdigit())
            if len(digits) >= 12:
                result['aadhaar_number'] = digits[:12]

        # DOB
        elif any(keyword in field_lower for keyword in dob_keywords):
            result['dob'] = field.split(':')[-1].strip()
            if not result['dob'] and any(c.isdigit() for c in field):
                date_patterns = re.findall(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}', field)
                if date_patterns:
                    result['dob'] = date_patterns[0]

        # Gender
        elif 'gender' in field_lower or 'sex' in field_lower:
            if 'f' in field_lower or 'female' in field_lower:
                result['gender'] = 'Female'
            elif 'm' in field_lower or 'male' in field_lower:
                result['gender'] = 'Male'
            else:
                result['gender'] = field.split(':')[-1].strip()

        # Mobile
        elif any(keyword in field_lower for keyword in ['mobile', 'phone', 'mob']):
            digits = ''.join(c for c in field if c.isdigit())
            if len(digits) >= 10:
                result['mobile'] = digits[-10:]

        # Address
        elif any(keyword in field_lower for keyword in ['address', 'addr', 'loc', 'location']):
            result['address'] = field.split(':')[-1].strip()

    return result


def parse_smartcard_data(data):
    """Parses Smart Card QR text data into structured dictionary."""
    result = {
        "card_number": "",
        "card_type": "",
        "head_of_family": "",
        "address": "",
        "family_members": "",
        "issued_by": "",
        "aadhaar_link_status": ""
    }

    for field in data.split('\n'):
        field_lower = field.lower()

        if 'card number' in field_lower:
            result['card_number'] = field.split(':')[-1].strip()
        elif 'type' in field_lower:
            result['card_type'] = field.split(':')[-1].strip()
        elif 'head' in field_lower:
            result['head_of_family'] = field.split(':')[-1].strip()
        elif 'address' in field_lower:
            result['address'] = field.split(':')[-1].strip()
        elif 'members' in field_lower:
            result['family_members'] = field.split(':')[-1].strip()
        elif 'issued' in field_lower:
            result['issued_by'] = field.split(':')[-1].strip()
        elif 'linked' in field_lower:
            result['aadhaar_link_status'] = field.split(':')[-1].strip()

    return result


def save_session_data(session, aadhaar_data, smart_data):
    """Saves required fields into Flask session."""
    session['user_name'] = aadhaar_data.get('user_name', '')
    session['dob'] = aadhaar_data.get('dob', '')
    session['gender'] = aadhaar_data.get('gender', '')
    session['address'] = aadhaar_data.get('address', '')
    session['aadhaar_number'] = aadhaar_data.get('aadhaar_number', '')
    session['card_number'] = smart_data.get('card_number', '')
    session['card_type'] = smart_data.get('card_type', '')
    session['mobile'] = aadhaar_data.get('mobile', '')