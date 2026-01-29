"""
Excel Upload Routes
==================
Handles Excel file uploads for patient data with:
1. Field mapping to database schema
2. Duplicate detection using Case Linkage
3. Case scoring for new patients
4. Automatic follow-up initiation (email + WhatsApp)
"""

import os
import pandas as pd
from flask import Blueprint, request, jsonify, current_app, session
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile

# Import models and services
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import db, Patient, User, Drug
from pv_backend.services.case_matching import CaseMatchingEngine
from pv_backend.services.case_scoring import evaluate_case, score_case, check_followup
from pv_backend.services.followup_agent import FollowupAgent
from pv_backend.routes.followup_routes import store_followup_token

excel_upload_bp = Blueprint('excel_upload', __name__, url_prefix='/api/excel')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Field mapping: Excel column names -> Patient model fields
FIELD_MAPPING = {
    # Common variations of column names
    'patient_name': 'name',
    'patient name': 'name',
    'name': 'name',
    'full_name': 'name',
    'full name': 'name',
    
    'patient_email': 'email',
    'patient email': 'email',
    'email': 'email',
    'email_address': 'email',
    'email address': 'email',
    
    'patient_phone': 'phone',
    'patient phone': 'phone',
    'phone': 'phone',
    'phone_number': 'phone',
    'phone number': 'phone',
    'mobile': 'phone',
    'contact': 'phone',
    
    'patient_age': 'age',
    'patient age': 'age',
    'age': 'age',
    
    'patient_gender': 'gender',
    'patient gender': 'gender',
    'gender': 'gender',
    'sex': 'gender',
    
    'drug_name': 'drug_name',
    'drug name': 'drug_name',
    'drug': 'drug_name',
    'medication': 'drug_name',
    'medicine': 'drug_name',
    'medicine_name': 'drug_name',
    'medicine name': 'drug_name',
    
    'symptoms': 'symptoms',
    'symptom': 'symptoms',
    'side_effects': 'symptoms',
    'side effects': 'symptoms',
    'adverse_effects': 'symptoms',
    'adverse effects': 'symptoms',
    'reaction': 'symptoms',
    'adverse_reaction': 'symptoms',
    'adverse reaction': 'symptoms',
    'description': 'symptoms',  # Map description to symptoms
    
    'risk_level': 'risk_level',
    'risk level': 'risk_level',
    'severity': 'risk_level',
    'risk': 'risk_level',
    
    'case_status': 'case_status',
    'case status': 'case_status',
    'status': 'case_status',
    'outcome': 'case_status',  # Map outcome to case_status
    
    'symptom_onset_date': 'symptom_onset_date',
    'symptom onset date': 'symptom_onset_date',
    'onset_date': 'symptom_onset_date',
    'onset date': 'symptom_onset_date',
    'start_date': 'symptom_onset_date',
    'start date': 'symptom_onset_date',
    
    'reporter_type': 'reporter_type',
    'reporter type': 'reporter_type',
    'reported_by': 'reporter_type',
    'reported by': 'reporter_type',
    
    # Additional fields from Excel template
    'drug_code': 'drug_code',
    'drug code': 'drug_code',
    'batch': 'drug_code',
    'batch_code': 'drug_code',
    'batch code': 'drug_code',
    
    'dosage': 'dosage',
    'dose': 'dosage',
    
    'route': 'route',
    'administration_route': 'route',
    'administration route': 'route',
    
    'indication': 'indication',
    'reason': 'indication',
    
    'consent': 'consent',
}


def map_excel_columns(df):
    """
    Map Excel column names to database field names.
    Returns DataFrame with standardized column names.
    """
    column_mapping = {}
    unmapped_columns = []
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in FIELD_MAPPING:
            column_mapping[col] = FIELD_MAPPING[col_lower]
        else:
            unmapped_columns.append(col)
    
    # Rename columns
    df_mapped = df.rename(columns=column_mapping)
    
    return df_mapped, unmapped_columns


def check_duplicate_patient(name, drug_name, age, gender, symptoms=None, phone=None, email=None):
    """
    Check for duplicate patient entries using Case Matching Engine.
    
    Returns:
        dict: {
            'is_duplicate': bool,
            'action': 'ACCEPT' | 'LINK' | 'REJECT',
            'existing_case': Patient object if duplicate found,
            'match_score': float,
            'reason': str
        }
    """
    # Prepare the new case data
    new_case = {
        'name': name,
        'drug_name': drug_name or 'Not Specified',
        'age': age,
        'gender': gender,
        'symptoms': symptoms or '',
        'phone': phone,
        'email': email
    }
    
    # Find existing patients with similar drug
    existing_patients = Patient.query.filter(
        Patient.drug_name.ilike(f"%{drug_name}%") if drug_name else True
    ).all()
    
    if not existing_patients:
        return {
            'is_duplicate': False,
            'action': 'ACCEPT',
            'existing_case': None,
            'match_score': 0,
            'reason': 'No existing patients with this drug'
        }
    
    # Use Case Matching Engine
    engine = CaseMatchingEngine(threshold=0.85)
    
    best_match = None
    best_score = 0
    
    for existing in existing_patients:
        # Skip if different drug (case insensitive)
        if drug_name and existing.drug_name:
            if drug_name.lower().strip() != existing.drug_name.lower().strip():
                continue
        
        result = engine.calculate_case_similarity(new_case, existing)
        
        # Check for exact name match
        name_match = False
        if name and existing.name:
            name_similarity = engine.calculate_text_similarity(name, existing.name)
            name_match = name_similarity >= 0.9
        
        # Check phone/email match for identity confirmation
        identity_match = False
        if phone and existing.phone and phone == existing.phone:
            identity_match = True
        if email and existing.email and email.lower() == existing.email.lower():
            identity_match = True
        
        # Calculate combined score
        combined_score = result['similarity_score']
        if name_match:
            combined_score = min(1.0, combined_score + 0.2)
        if identity_match:
            combined_score = min(1.0, combined_score + 0.3)
        
        if combined_score > best_score:
            best_score = combined_score
            best_match = {
                'patient': existing,
                'score': combined_score,
                'name_match': name_match,
                'identity_match': identity_match
            }
    
    if best_match and best_score >= 0.95:
        # Very high match - exact duplicate
        return {
            'is_duplicate': True,
            'action': 'REJECT',
            'existing_case': best_match['patient'],
            'match_score': best_score,
            'reason': f"Exact duplicate - Patient '{best_match['patient'].name}' with drug '{best_match['patient'].drug_name}' already exists (ID: {best_match['patient'].id})"
        }
    elif best_match and best_score >= 0.85:
        # High match - should be case-linked
        return {
            'is_duplicate': True,
            'action': 'LINK',
            'existing_case': best_match['patient'],
            'match_score': best_score,
            'reason': f"Similar case found - Will link to case {best_match['patient'].id} (Match: {best_score:.1%})"
        }
    else:
        return {
            'is_duplicate': False,
            'action': 'ACCEPT',
            'existing_case': best_match['patient'] if best_match else None,
            'match_score': best_score,
            'reason': 'No duplicate detected - accepting as new case'
        }


def send_followup_to_patient(patient):
    """
    Send follow-up via email and/or WhatsApp if contact info available.
    Returns result dict with channels sent.
    """
    try:
        agent = FollowupAgent()
        token = agent.generate_followup_token(patient.id)
        
        # Store token for validation
        store_followup_token(patient.id, token)
        
        result = {
            'patient_id': patient.id,
            'channels_sent': [],
            'errors': []
        }
        
        # Send email if available
        if patient.email:
            email_result = agent.send_followup_email(patient, token)
            if email_result.get('success'):
                result['channels_sent'].append('email')
            else:
                result['errors'].append(f"Email: {email_result.get('error', 'Failed')}")
        
        # Send WhatsApp if phone available
        if patient.phone:
            whatsapp_result = agent.send_followup_whatsapp(patient, token)
            if whatsapp_result.get('success'):
                result['channels_sent'].append('whatsapp')
            else:
                result['errors'].append(f"WhatsApp: {whatsapp_result.get('error', 'Failed')}")
        
        # Update patient follow-up status
        if result['channels_sent']:
            patient.followup_sent_date = datetime.utcnow()
            patient.follow_up_sent = True
            patient.followup_pending = True
            db.session.commit()
        
        result['success'] = len(result['channels_sent']) > 0
        return result
        
    except Exception as e:
        return {
            'success': False,
            'patient_id': patient.id,
            'channels_sent': [],
            'errors': [str(e)]
        }


def generate_patient_id():
    """Generate a unique patient ID"""
    import random
    while True:
        patient_id = f"PT-{random.randint(10000, 99999)}"
        if not Patient.query.filter_by(id=patient_id).first():
            return patient_id


@excel_upload_bp.route('/upload', methods=['POST'])
def upload_excel():
    """
    Upload Excel file with patient data.
    
    Process:
    1. Parse Excel file and map fields
    2. For each row:
       a. Check for duplicates using Case Linkage
       b. If REJECT: skip (exact duplicate)
       c. If LINK: add with link to existing case
       d. If ACCEPT: add as new case
    3. Score each new patient record
    4. Send follow-up email/WhatsApp if contact info available
    
    Returns:
        JSON with upload results including:
        - total_rows: Number of rows in Excel
        - imported: Number successfully imported
        - duplicates_rejected: Number of exact duplicates skipped
        - duplicates_linked: Number linked to existing cases
        - followups_sent: Number of follow-ups initiated
        - errors: List of any errors encountered
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        # Read Excel file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Map columns to database fields
        df_mapped, unmapped_columns = map_excel_columns(df)
        
        # Initialize results
        results = {
            'total_rows': len(df),
            'imported': 0,
            'duplicates_rejected': 0,
            'duplicates_linked': 0,
            'followups_sent': 0,
            'scoring_completed': 0,
            'errors': [],
            'unmapped_columns': unmapped_columns,
            'imported_patients': [],
            'rejected_duplicates': [],
            'linked_cases': [],
            'followup_results': []
        }
        
        # Get current user (if authenticated via session)
        current_user = None
        created_by = None
        if 'user_id' in session:
            current_user = User.query.get(session['user_id'])
            if current_user:
                created_by = current_user.id
        
        # Process each row
        for idx, row in df_mapped.iterrows():
            row_num = idx + 2  # Excel rows are 1-indexed + header
            
            try:
                # Extract patient data with defaults
                name = str(row.get('name', '')).strip() if pd.notna(row.get('name')) else None
                email = str(row.get('email', '')).strip() if pd.notna(row.get('email')) else None
                phone = str(row.get('phone', '')).strip() if pd.notna(row.get('phone')) else None
                
                # Handle age - convert to int, default to 0 if missing
                age_val = row.get('age')
                if pd.notna(age_val):
                    try:
                        age = int(float(str(age_val).strip()))
                    except (ValueError, TypeError):
                        age = 0
                else:
                    age = 0
                
                # Handle gender - default to 'Unknown' if missing
                gender = str(row.get('gender', '')).strip() if pd.notna(row.get('gender')) else 'Unknown'
                if not gender:
                    gender = 'Unknown'
                
                drug_name = str(row.get('drug_name', '')).strip() if pd.notna(row.get('drug_name')) else None
                symptoms = str(row.get('symptoms', '')).strip() if pd.notna(row.get('symptoms')) else None
                risk_level = str(row.get('risk_level', 'Medium')).strip() if pd.notna(row.get('risk_level')) else 'Medium'
                case_status = str(row.get('case_status', 'Active')).strip() if pd.notna(row.get('case_status')) else 'Active'
                
                # Validate required fields
                if not name:
                    results['errors'].append(f"Row {row_num}: Missing patient name")
                    continue
                
                if not drug_name:
                    results['errors'].append(f"Row {row_num}: Missing drug name")
                    continue
                
                # =====================================
                # STEP 1: Check for duplicates using Case Linkage
                # =====================================
                duplicate_check = check_duplicate_patient(
                    name=name,
                    drug_name=drug_name,
                    age=age,
                    gender=gender,
                    symptoms=symptoms,
                    phone=phone,
                    email=email
                )
                
                if duplicate_check['action'] == 'REJECT':
                    # Exact duplicate - skip this row
                    results['duplicates_rejected'] += 1
                    results['rejected_duplicates'].append({
                        'row': row_num,
                        'name': name,
                        'drug': drug_name,
                        'existing_patient_id': duplicate_check['existing_case'].id,
                        'match_score': duplicate_check['match_score'],
                        'reason': duplicate_check['reason']
                    })
                    continue
                
                # =====================================
                # STEP 2: Create patient record
                # =====================================
                patient_id = generate_patient_id()
                
                patient = Patient(
                    id=patient_id,
                    name=name,
                    email=email,
                    phone=phone,
                    age=age,
                    gender=gender,
                    drug_name=drug_name,
                    symptoms=symptoms,
                    risk_level=risk_level,
                    case_status=case_status,
                    created_by=created_by,
                    created_at=datetime.utcnow()
                )
                
                # If LINK action, link to existing case
                if duplicate_check['action'] == 'LINK':
                    existing_case = duplicate_check['existing_case']
                    patient.linked_case_id = existing_case.id
                    patient.match_score = duplicate_check['match_score']
                    patient.match_notes = duplicate_check['reason']
                    
                    results['duplicates_linked'] += 1
                    results['linked_cases'].append({
                        'row': row_num,
                        'new_patient_id': patient_id,
                        'linked_to': existing_case.id,
                        'match_score': duplicate_check['match_score']
                    })
                
                db.session.add(patient)
                
                # Link patient to the current doctor
                if current_user and current_user.role == 'doctor':
                    patient.doctors.append(current_user)
                
                db.session.commit()
                
                # =====================================
                # STEP 3: Score the patient record
                # =====================================
                try:
                    evaluate_case(patient)
                    score_case(patient)
                    check_followup(patient)
                    db.session.commit()
                    results['scoring_completed'] += 1
                except Exception as e:
                    results['errors'].append(f"Row {row_num}: Scoring failed - {str(e)}")
                
                # =====================================
                # STEP 4: Send follow-up if contact available
                # =====================================
                if email or phone:
                    followup_result = send_followup_to_patient(patient)
                    results['followup_results'].append({
                        'patient_id': patient_id,
                        'row': row_num,
                        'channels': followup_result.get('channels_sent', []),
                        'success': followup_result.get('success', False),
                        'errors': followup_result.get('errors', [])
                    })
                    
                    if followup_result.get('success'):
                        results['followups_sent'] += 1
                
                results['imported'] += 1
                results['imported_patients'].append({
                    'row': row_num,
                    'patient_id': patient_id,
                    'name': name,
                    'drug': drug_name,
                    'linked': duplicate_check['action'] == 'LINK'
                })
                
            except Exception as e:
                results['errors'].append(f"Row {row_num}: {str(e)}")
                continue
        
        # Cleanup temp file
        try:
            os.remove(filepath)
            os.rmdir(temp_dir)
        except:
            pass
        
        # Summary
        results['summary'] = {
            'success_rate': f"{(results['imported'] / results['total_rows'] * 100):.1f}%" if results['total_rows'] > 0 else "0%",
            'duplicate_rejection_rate': f"{(results['duplicates_rejected'] / results['total_rows'] * 100):.1f}%" if results['total_rows'] > 0 else "0%",
            'followup_rate': f"{(results['followups_sent'] / results['imported'] * 100):.1f}%" if results['imported'] > 0 else "0%"
        }
        
        return jsonify({
            'success': True,
            'message': f"Import completed: {results['imported']}/{results['total_rows']} patients imported",
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to process file: {str(e)}'
        }), 500


@excel_upload_bp.route('/template', methods=['GET'])
def get_template_info():
    """
    Get information about expected Excel template format.
    """
    return jsonify({
        'required_columns': ['name', 'drug_name'],
        'optional_columns': [
            'email', 'phone', 'age', 'gender', 'symptoms', 
            'risk_level', 'case_status', 'symptom_onset_date'
        ],
        'column_aliases': FIELD_MAPPING,
        'example_row': {
            'name': 'John Doe',
            'email': 'john@example.com',
            'phone': '+919876543210',
            'age': 45,
            'gender': 'Male',
            'drug_name': 'Aspirin',
            'symptoms': 'Headache, Nausea',
            'risk_level': 'Medium',
            'case_status': 'Active'
        },
        'notes': [
            'File formats supported: xlsx, xls, csv',
            'Column names are case-insensitive',
            'Duplicate patients (same name + same drug) will be rejected',
            'Similar patients will be linked to existing cases',
            'Follow-ups sent automatically if email/phone provided'
        ]
    }), 200


@excel_upload_bp.route('/preview', methods=['POST'])
def preview_upload():
    """
    Preview Excel file before importing.
    Shows field mapping and potential duplicates without actually importing.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        # Read Excel file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Map columns
        df_mapped, unmapped_columns = map_excel_columns(df)
        
        # Preview results
        preview = {
            'total_rows': len(df),
            'columns_found': list(df.columns),
            'columns_mapped': {col: FIELD_MAPPING.get(col.lower().strip(), 'unmapped') for col in df.columns},
            'unmapped_columns': unmapped_columns,
            'sample_data': df_mapped.head(5).to_dict(orient='records'),
            'potential_duplicates': [],
            'potential_links': [],
            'ready_to_import': []
        }
        
        # Check each row for potential duplicates
        for idx, row in df_mapped.iterrows():
            if idx >= 10:  # Only check first 10 rows for preview
                break
                
            name = str(row.get('name', '')).strip() if pd.notna(row.get('name')) else None
            drug_name = str(row.get('drug_name', '')).strip() if pd.notna(row.get('drug_name')) else None
            
            if name and drug_name:
                duplicate_check = check_duplicate_patient(
                    name=name,
                    drug_name=drug_name,
                    age=int(row.get('age', 0)) if pd.notna(row.get('age')) else None,
                    gender=str(row.get('gender', '')).strip() if pd.notna(row.get('gender')) else None,
                    symptoms=str(row.get('symptoms', '')).strip() if pd.notna(row.get('symptoms')) else None,
                    phone=str(row.get('phone', '')).strip() if pd.notna(row.get('phone')) else None,
                    email=str(row.get('email', '')).strip() if pd.notna(row.get('email')) else None
                )
                
                row_info = {
                    'row': idx + 2,
                    'name': name,
                    'drug': drug_name,
                    'action': duplicate_check['action'],
                    'match_score': duplicate_check['match_score'],
                    'reason': duplicate_check['reason']
                }
                
                if duplicate_check['action'] == 'REJECT':
                    preview['potential_duplicates'].append(row_info)
                elif duplicate_check['action'] == 'LINK':
                    preview['potential_links'].append(row_info)
                else:
                    preview['ready_to_import'].append(row_info)
        
        # Cleanup
        try:
            os.remove(filepath)
            os.rmdir(temp_dir)
        except:
            pass
        
        return jsonify({
            'success': True,
            'preview': preview
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to preview file: {str(e)}'
        }), 500
