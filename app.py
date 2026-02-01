"""
Inteleyzer - Pharmacovigilance Platform
Main Flask Application Entry Point
Supports: Pharmaceutical Companies, Doctors, and Local Pharmacies
Run on: http://127.0.0.1:5000
"""

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv(override=True)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from models import db, User, Patient, Drug, Alert, CaseAgent, FollowUp, SideEffectReport, AgentFollowupTracking, hospital_doctor, hospital_drug, hospital_pharmacy, doctor_patient
from pv_backend.services.case_matching import match_new_case, should_accept_case
from pv_backend.services.case_scoring import CaseScoringEngine, evaluate_case, score_case, check_followup
from pv_backend.services.quality_agent import QualityAgentOrchestrator, FollowUpManager
from pv_backend.routes.followup_routes import init_followup_routes, store_followup_token
from pv_backend.services.followup_agent import FollowupAgent
from pv_backend.routes.excel_routes import excel_upload_bp
from pv_backend.routes.pharmacy_report_routes import pharmacy_report_bp
# Import pharmacy report models so tables are created
from pv_backend.models.pharmacy_report import PharmacyReport, AnonymousReport, IdentifiedReport, AggregatedReport
from auth_config import JWTConfig, token_required, session_required, SESSION_TIMEOUT_MINUTES, TOKEN_EXPIRY_HOURS
import os
import random
from datetime import datetime, timedelta

app = Flask(__name__)


def auto_send_followup(patient):
    """
    Automatically start PV Agent follow-up cycle for a patient.
    Uses PVAgentOrchestrator for Day 1/3/5/7 scheduling with:
    - Case scoring
    - LLM-generated questions
    - Per-day email/WhatsApp coordination
    
    Args:
        patient: Patient model instance with email and/or phone field
        
    Returns:
        dict: Result containing success status and tracking info
    """
    if not patient:
        return {'success': False, 'reason': 'no_patient', 'message': 'No patient provided'}
    
    # Check if patient has any contact method
    has_email = bool(patient.email)
    has_phone = bool(patient.phone)
    
    if not has_email and not has_phone:
        return {
            'success': False,
            'reason': 'no_contact',
            'message': 'Patient has no email address or phone number'
        }
    
    try:
        # Use PV Agent Orchestrator for enhanced follow-up
        from pv_backend.services.followup_agent import PVAgentOrchestrator
        
        orchestrator = PVAgentOrchestrator()
        result = orchestrator.start_tracking(patient)
        
        if result.get('success'):
            # Update patient record
            patient.followup_sent_date = datetime.utcnow()
            patient.followup_pending = True
            patient.follow_up_sent = True
            db.session.commit()
            
            print(f"[OK] PV Agent started for patient {patient.id} - Day 1 of 1/3/5/7 cycle")
            
            return {
                'success': True,
                'patient_id': patient.id,
                'tracking_id': result.get('tracking_id'),
                'current_day': result.get('current_day', 1),
                'email': result.get('send_result', {}).get('email'),
                'whatsapp': result.get('send_result', {}).get('whatsapp'),
                'questions': result.get('questions', []),
                'message': 'PV Agent follow-up cycle started (Day 1/3/5/7)'
            }
        else:
            return {
                'success': False,
                'reason': 'orchestrator_error',
                'message': result.get('error', 'Failed to start PV Agent tracking'),
                'patient_id': patient.id
            }
            
    except Exception as e:
        print(f"[ERROR] Exception in PV Agent: {str(e)}")
        # Fallback to basic follow-up
        try:
            agent = FollowupAgent()
            token = agent.generate_followup_token(patient.id)
            store_followup_token(patient.id, token)
            
            results = {'success': False, 'patient_id': patient.id, 'channels_sent': 0}
            
            if has_email:
                email_result = agent.send_followup_email(patient, token)
                if email_result.get('success'):
                    results['channels_sent'] += 1
            
            if has_phone:
                whatsapp_result = agent.send_followup_whatsapp(patient, token)
                if whatsapp_result.get('success'):
                    results['channels_sent'] += 1
            
            if results['channels_sent'] > 0:
                patient.followup_sent_date = datetime.utcnow()
                patient.followup_pending = True
                patient.follow_up_sent = True
                db.session.commit()
                results['success'] = True
                results['message'] = f"Fallback: Follow-up sent via {results['channels_sent']} channel(s)"
            
            return results
            
        except Exception as fallback_error:
            return {
                'success': False,
                'reason': 'exception',
                'message': str(e),
                'patient_id': patient.id if patient else None
            }


def auto_send_followup_email(patient):
    """
    Automatically send follow-up email to patient immediately after creation.
    Only sends if patient has a valid email address.
    
    DEPRECATED: Use auto_send_followup() for multi-channel support.
    
    Args:
        patient: Patient model instance with email field
        
    Returns:
        dict: Result containing success status and details
    """
    # Use the new multi-channel function
    result = auto_send_followup(patient)
    
    # Return in old format for backward compatibility
    if result.get('email', {}).get('sent'):
        return {
            'success': True,
            'message': f"Follow-up email sent to {patient.email}",
            'email': patient.email,
            'patient_id': patient.id
        }
    elif result.get('whatsapp', {}).get('sent'):
        # If email failed but WhatsApp succeeded, still return success
        return {
            'success': True,
            'message': f"Follow-up sent via WhatsApp to {patient.phone}",
            'phone': patient.phone,
            'patient_id': patient.id
        }
    else:
        return {
            'success': result.get('success', False),
            'reason': result.get('reason', 'send_failed'),
            'message': result.get('message', 'Failed to send follow-up'),
            'email': patient.email if patient and patient.email else None
        }


def check_duplicate_patient(name, drug_name, age, gender, symptoms=None, phone=None, email=None):
    """
    Check for duplicate patient entries before adding to database.
    Uses case matching to detect potential duplicates for same patient + same drug.
    
    Args:
        name: Patient name
        drug_name: Medication/drug name
        age: Patient age
        gender: Patient gender
        symptoms: Patient symptoms (optional)
        phone: Patient phone (optional)
        email: Patient email (optional)
        
    Returns:
        dict: {
            'is_duplicate': bool,
            'action': 'ACCEPT' | 'LINK' | 'REJECT',
            'existing_case': Patient object if duplicate found,
            'match_score': float,
            'reason': str
        }
    """
    from pv_backend.services.case_matching import CaseMatchingEngine
    
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
    
    # Find existing patients with similar characteristics
    # First, check for exact same drug + similar demographics
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
    
    # Use Case Matching Engine with high threshold for duplicates
    engine = CaseMatchingEngine(threshold=0.85)
    
    best_match = None
    best_score = 0
    
    for existing in existing_patients:
        # Skip if different drug (case insensitive)
        if drug_name and existing.drug_name:
            if drug_name.lower().strip() != existing.drug_name.lower().strip():
                continue
        
        result = engine.calculate_case_similarity(new_case, existing)
        
        # Check for exact duplicate (same name + same drug + similar age/gender)
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
                'identity_match': identity_match,
                'breakdown': result['breakdown']
            }
    
    if best_match and best_score >= 0.95:
        # Very high match - likely exact duplicate (reject)
        return {
            'is_duplicate': True,
            'action': 'REJECT',
            'existing_case': best_match['patient'],
            'match_score': best_score,
            'reason': f"Exact duplicate detected - Patient '{best_match['patient'].name}' with same drug '{best_match['patient'].drug_name}' already exists (ID: {best_match['patient'].id})"
        }
    elif best_match and best_score >= 0.85:
        # High match - should be case-linked
        return {
            'is_duplicate': True,
            'action': 'LINK',
            'existing_case': best_match['patient'],
            'match_score': best_score,
            'reason': f"Similar case found - Will link to existing case {best_match['patient'].id} (Match: {best_score:.1%})"
        }
    else:
        # No significant match - accept as new
        return {
            'is_duplicate': False,
            'action': 'ACCEPT',
            'existing_case': best_match['patient'] if best_match else None,
            'match_score': best_score,
            'reason': 'No duplicate detected - accepting as new case'
        }


app.config['SECRET_KEY'] = 'inteleyzer-secret-key-dev'

# Template configuration
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Session configuration
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

# Get absolute path for database
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
os.makedirs(instance_path, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "inteleyzer.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db.init_app(app)

# Register Excel upload blueprint
app.register_blueprint(excel_upload_bp)

# Register Pharmacy Reports blueprint
app.register_blueprint(pharmacy_report_bp)

# Add cache-busting headers for development
@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Create database tables
with app.app_context():
    db.create_all()

# --- UI Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

# Doctor Routes
@app.route('/doctor/dashboard')
def doctor_dashboard():
    return render_template('doctor/dashboard.html')

@app.route('/doctor/patients')
def doctor_patients():
    return render_template('doctor/patients.html')

@app.route('/doctor/alerts')
def doctor_alerts():
    return render_template('doctor/alerts.html')

@app.route('/doctor/warnings')
def doctor_warnings():
    return render_template('doctor/warnings.html')

@app.route('/doctor/report')
def doctor_report():
    return render_template('doctor/report.html')

@app.route('/doctor/settings')
def doctor_settings():
    return render_template('doctor/settings.html')

# Pharma Routes
@app.route('/pharma/dashboard')
def pharma_dashboard():
    return render_template('pharma/dashboard.html')

@app.route('/pharma/drugs')
def pharma_drugs():
    return render_template('pharma/drugs.html')

@app.route('/pharma/reports')
def pharma_reports():
    return render_template('pharma/reports.html')

@app.route('/pharma/analysis')
def pharma_analysis():
    return render_template('pharma/analysis.html')

# Pharmacy Routes
@app.route('/pharmacy/dashboard')
def pharmacy_dashboard():
    return render_template('pharmacy/dashboard.html')

@app.route('/pharmacy/reports')
def pharmacy_reports():
    return render_template('pharmacy/reports.html')

@app.route('/pharmacy/report')
def pharmacy_report():
    return render_template('pharmacy/report.html')

@app.route('/pharmacy/alerts')
def pharmacy_alerts():
    return render_template('pharmacy/alerts.html')

@app.route('/pharmacy/history')
def pharmacy_history():
    return render_template('pharmacy/history.html')

# --- API Routes ---

# Authentication APIs
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    existing = User.query.filter_by(email=data['email']).first()
    if existing:
        return jsonify({'success': False, 'message': 'Email already registered'})
    
    user = User(
        name=data['name'],
        email=data['email'],
        password=data['password'],
        role=data['role']
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email'], password=data['password']).first()
    
    if user:
        # Generate JWT token
        token = JWTConfig.generate_token(user.id, user.email, user.role, user.name)
        
        # Set session with expiry tracking
        session.permanent = True
        session['user_id'] = user.id
        session['role'] = user.role
        session['user_name'] = user.name
        session['user_email'] = user.email
        session['session_start'] = datetime.utcnow()
        session['token'] = token
        
        # For hospital role, set hospital_name from user name
        if user.role == 'hospital':
            session['hospital_name'] = user.name
        
        return jsonify({
            'success': True, 
            'user': {
                'id': user.id, 
                'name': user.name, 
                'role': user.role,
                'email': user.email
            },
            'token': token,
            'token_expiry': TOKEN_EXPIRY_HOURS,
            'session_timeout': SESSION_TIMEOUT_MINUTES
        })
    
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/api/auth/logout', methods=['POST'])
def logout_api():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/auth/refresh-token', methods=['POST'])
def refresh_token():
    """Refresh JWT token before expiry"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    # Generate new token
    token = JWTConfig.generate_token(
        session['user_id'],
        session['user_email'],
        session['role'],
        session['user_name']
    )
    
    session['token'] = token
    session['session_start'] = datetime.utcnow()
    
    return jsonify({
        'success': True,
        'token': token,
        'token_expiry': TOKEN_EXPIRY_HOURS,
        'session_timeout': SESSION_TIMEOUT_MINUTES
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/api/auth/me')
def get_current_user():
    if 'user_id' in session:
        return jsonify({
            'id': session['user_id'],
            'name': session['user_name'],
            'role': session['role']
        })
    return jsonify(None), 401

# Patient/Report APIs
@app.route('/api/patients', methods=['GET'])
def get_patients():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    
    if user.role == 'pharma':
        company_drugs = [d.name for d in Drug.query.filter_by(company_id=user.id).all()]
        patients = Patient.query.filter(Patient.drug_name.in_(company_drugs)).all() if company_drugs else []
    elif user.role == 'doctor':
        patients = Patient.query.filter(Patient.doctors.contains(user)).all()
    else:
        patients = []
    
    # Return array directly for pharma.js compatibility
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'age': p.age,
        'gender': p.gender,
        'drugName': p.drug_name,
        'symptoms': p.symptoms,
        'riskLevel': p.risk_level,
        'createdAt': p.created_at.isoformat() if p.created_at else None,
        # Case Scoring summary
        'caseScore': p.case_score,
        'strengthLevel': p.strength_level,
        'followUpRequired': p.follow_up_required,
        'caseStatus': p.case_status
    } for p in patients])

@app.route('/api/patients', methods=['POST'])
def create_patient():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        mode = data.get('mode', 'identity')
        
        # Extract patient data based on mode
        if mode == 'identity':
            name = data.get('name')
            phone = data.get('contactDetails')
            email = data.get('email')
            age = int(data.get('age'))
            gender = data.get('gender')
            drug_name = data.get('medication') or 'Not Specified'
            symptoms = data.get('symptoms', '')
            risk_level = data.get('riskLevel', 'Low')
        else:  # anonymized mode
            name = 'Anonymized Patient'
            phone = None
            email = None
            age_group = data.get('ageGroup', '31-45')
            if '-' in age_group:
                age_parts = age_group.split('-')
                age = (int(age_parts[0]) + int(age_parts[1])) // 2
            else:
                age = 35
            gender = data.get('gender')
            drug_name = 'Not Specified'
            symptoms = data.get('notes', '')
            risk_level = data.get('riskCategory', 'Low')
        
        # === DUPLICATE CHECK ===
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
            # Exact duplicate - reject the entry
            existing = duplicate_check['existing_case']
            return jsonify({
                'success': False,
                'duplicate': True,
                'message': duplicate_check['reason'],
                'existing_patient': {
                    'id': existing.id,
                    'name': existing.name,
                    'drug_name': existing.drug_name,
                    'created_at': existing.created_at.isoformat() if existing.created_at else None
                },
                'match_score': duplicate_check['match_score']
            }), 409  # Conflict status code
        
        elif duplicate_check['action'] == 'LINK':
            # Similar case found - create new entry but link to existing case
            existing = duplicate_check['existing_case']
            patient_id = data.get('patientId') if mode == 'identity' else data.get('anonId')
            if not patient_id:
                patient_id = f"PT-{random.randint(10000, 99999)}"
                while Patient.query.get(patient_id):
                    patient_id = f"PT-{random.randint(10000, 99999)}"
            
            patient = Patient(
                id=patient_id,
                name=name,
                phone=phone,
                email=email,
                age=age,
                gender=gender,
                drug_name=drug_name,
                symptoms=symptoms,
                risk_level=risk_level,
                case_status='Linked',
                linked_case_id=existing.id,
                match_score=duplicate_check['match_score'],
                match_notes=f"Auto-linked: {duplicate_check['reason']}",
                created_by=session['user_id']
            )
        else:
            # No duplicate - create new patient
            patient_id = data.get('patientId') if mode == 'identity' else data.get('anonId')
            if not patient_id:
                patient_id = f"PT-{random.randint(10000, 99999)}"
                while Patient.query.get(patient_id):
                    patient_id = f"PT-{random.randint(10000, 99999)}"
            
            patient = Patient(
                id=patient_id,
                name=name,
                phone=phone,
                email=email,
                age=age,
                gender=gender,
                drug_name=drug_name,
                symptoms=symptoms,
                risk_level=risk_level,
                case_status='Active',
                created_by=session['user_id']
            )
        
        db.session.add(patient)
        
        # Link patient to doctor if user is a doctor
        user = User.query.get(session['user_id'])
        if user.role == 'doctor':
            db.session.execute(
                doctor_patient.insert().values(
                    doctor_id=user.id,
                    patient_id=patient.id
                )
            )
        
        db.session.commit()
        
        # Auto-start PV Agent follow-up cycle if patient has email or phone
        followup_result = None
        if patient.email or patient.phone:
            followup_result = auto_send_followup(patient)
            print(f"📱 PV Agent result for {patient.id}: {followup_result}")
        
        response_data = {
            'success': True,
            'patient': {
                'id': patient.id,
                'name': patient.name,
                'email': patient.email,
                'age': patient.age,
                'gender': patient.gender,
                'drug_name': patient.drug_name,
                'symptoms': patient.symptoms,
                'risk_level': patient.risk_level,
                'case_status': patient.case_status,
                'linked_case_id': patient.linked_case_id,
                'created_at': patient.created_at.isoformat() if patient.created_at else datetime.utcnow().isoformat()
            },
            'duplicate_check': {
                'action': duplicate_check['action'],
                'match_score': duplicate_check['match_score'],
                'reason': duplicate_check['reason']
            }
        }
        
        # Include follow-up email result if email was sent
        if followup_result:
            response_data['followup_email'] = followup_result
        
        return jsonify(response_data)
    except Exception as e:
        db.session.rollback()
        print(f"Error creating patient: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404
    
    return jsonify({
        'success': True,
        'patient': {
            'id': patient.id,
            'name': patient.name,
            'phone': patient.phone,
            'email': patient.email,
            'age': patient.age,
            'gender': patient.gender,
            'drug_name': patient.drug_name,
            'symptoms': patient.symptoms,
            'risk_level': patient.risk_level,
            'created_at': patient.created_at.isoformat() if patient.created_at else None,
            'created_by': patient.created_by,
            # Case Linkage
            'case_status': patient.case_status,
            'linked_case_id': patient.linked_case_id,
            'match_score': patient.match_score,
            'match_notes': patient.match_notes,
            # Case Scoring
            'case_scoring': {
                'strength_level': patient.strength_level,
                'strength_score': patient.strength_score,
                'completeness_score': round(patient.completeness_score * 100, 1) if patient.completeness_score else 0,
                'temporal_clarity_score': round(patient.temporal_clarity_score * 100, 1) if patient.temporal_clarity_score else 0,
                'medical_confirmation_score': round(patient.medical_confirmation_score * 100, 1) if patient.medical_confirmation_score else 0,
                'followup_responsiveness_score': round(patient.followup_responsiveness_score * 100, 1) if patient.followup_responsiveness_score else 0,
                'case_score': patient.case_score,
                'polarity': patient.polarity,
                'case_score_interpretation': patient.case_score_interpretation,
                'follow_up_required': patient.follow_up_required,
                'evaluated_at': patient.evaluated_at.isoformat() if patient.evaluated_at else None
            },
            # Follow-up Email Status
            'followup_pending': patient.followup_pending if hasattr(patient, 'followup_pending') else False,
            'followup_completed': patient.followup_completed if hasattr(patient, 'followup_completed') else False,
            'followup_sent_date': patient.followup_sent_date.isoformat() if hasattr(patient, 'followup_sent_date') and patient.followup_sent_date else None
        }
    })


@app.route('/api/patients/<patient_id>', methods=['PUT'])
def update_patient(patient_id):
    """Update a patient record"""
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404
    
    data = request.get_json()
    
    # Update fields if provided
    if 'name' in data:
        patient.name = data['name']
    if 'phone' in data:
        patient.phone = data['phone']
    if 'email' in data:
        patient.email = data['email']
    if 'age' in data:
        patient.age = data['age']
    if 'gender' in data:
        patient.gender = data['gender']
    if 'drug_name' in data:
        patient.drug_name = data['drug_name']
    if 'symptoms' in data:
        patient.symptoms = data['symptoms']
    if 'risk_level' in data:
        patient.risk_level = data['risk_level']
    if 'side_effect' in data:
        patient.side_effect = data['side_effect']
    
    try:
        db.session.commit()
        return jsonify({'success': True, 'message': 'Patient updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/patients/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Delete a patient record"""
    from models import FollowupToken
    
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404
    
    try:
        # Delete related records first
        AgentFollowupTracking.query.filter_by(patient_id=patient_id).delete()
        FollowupToken.query.filter_by(patient_id=patient_id).delete()
        
        db.session.delete(patient)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Patient deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


# Stats APIs
@app.route('/api/stats')
def get_stats():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    
    if user.role == 'pharma':
        company_drugs = [d.name for d in Drug.query.filter_by(company_id=user.id).all()]
        patients = Patient.query.filter(Patient.drug_name.in_(company_drugs)).all() if company_drugs else []
        total_reports = len(patients)
        high_risk = len([p for p in patients if p.risk_level == 'High'])
        
        # Calculate distributions
        risk_dist = {
            'low': len([p for p in patients if p.risk_level == 'Low']),
            'medium': len([p for p in patients if p.risk_level == 'Medium']),
            'high': high_risk
        }
        
        gender_dist = {
            'male': len([p for p in patients if p.gender == 'Male']),
            'female': len([p for p in patients if p.gender == 'Female']),
            'other': len([p for p in patients if p.gender == 'Other'])
        }
        
        avg_age = sum([p.age for p in patients]) / len(patients) if patients else 0
        
        # Calculate Age Distribution
        age_dist = {
            '0-18': len([p for p in patients if p.age <= 18]),
            '19-40': len([p for p in patients if 19 <= p.age <= 40]),
            '41-60': len([p for p in patients if 41 <= p.age <= 60]),
            '60+': len([p for p in patients if p.age > 60])
        }
        
    elif user.role == 'doctor':
        patients = Patient.query.filter(Patient.doctors.contains(user)).all()
        total_reports = len(patients)
        high_risk = len([p for p in patients if p.risk_level == 'High'])
        risk_dist = {'low': 0, 'medium': 0, 'high': 0}
        gender_dist = {'male': 0, 'female': 0, 'other': 0}
        age_dist = {'0-18': 0, '19-40': 0, '41-60': 0, '60+': 0}
        avg_age = 0
    else:
        total_reports = 0
        high_risk = 0
        risk_dist = {'low': 0, 'medium': 0, 'high': 0}
        gender_dist = {'male': 0, 'female': 0, 'other': 0}
        age_dist = {'0-18': 0, '19-40': 0, '41-60': 0, '60+': 0}
        avg_age = 0
    
    return jsonify({
        'success': True,
        'totalReports': total_reports,
        'highRiskCount': high_risk,
        'avgAge': round(avg_age, 1),
        'riskDist': risk_dist,
        'genderDist': gender_dist,
        'ageDist': age_dist
    })

# Drug APIs
@app.route('/api/drugs', methods=['GET'])
def get_drugs():
    if 'user_id' not in session:
        return jsonify([]), 403
    
    user = User.query.get(session['user_id'])
    
    if user.role == 'pharma':
        drugs = Drug.query.filter_by(company_id=user.id).all()
    else:
        drugs = Drug.query.all()
    
    # Return array directly
    return jsonify([{
        'id': d.id,
        'name': d.name,
        'description': d.description,
        'activeIngredients': d.active_ingredients,
        'aiRiskAssessment': d.ai_risk_assessment,
        'aiRiskDetails': d.ai_risk_details,
        'createdAt': d.created_at.isoformat(),
        'companyName': d.company.name if d.company else None
    } for d in drugs])

@app.route('/api/drugs', methods=['POST'])
def add_drug():
    if 'user_id' not in session:
        return jsonify({'success': False}), 403
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharma':
        return jsonify({'success': False, 'message': 'Only pharma companies can add drugs'}), 403
    
    data = request.json
    drug = Drug(
        name=data['name'],
        company_id=user.id,
        description=data.get('description', ''),
        active_ingredients=data.get('activeIngredients', ''),
        ai_risk_assessment=data.get('riskAssessment', 'Medium'),
        ai_risk_details=data.get('riskDetails', 'Risk analysis pending')
    )
    
    db.session.add(drug)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'drug': {
            'id': drug.id,
            'name': drug.name,
            'description': drug.description,
            'activeIngredients': drug.active_ingredients,
            'aiRiskAssessment': drug.ai_risk_assessment,
            'aiRiskDetails': drug.ai_risk_details,
            'createdAt': drug.created_at.isoformat(),
            'companyName': user.name
        }
    })

# Alert APIs
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    alerts = Alert.query.order_by(Alert.created_at.desc()).limit(50).all()
    
    return jsonify({
        'success': True,
        'alerts': [{
            'id': a.id,
            'drug_name': a.drug_name,
            'message': a.message,
            'severity': a.severity,
            'created_at': a.created_at.isoformat(),
            'is_read': a.is_read,
            'sender': a.sender.name if a.sender else 'System'
        } for a in alerts]
    })

@app.route('/api/alerts', methods=['POST'])
def create_alert():
    if 'user_id' not in session:
        return jsonify({'success': False}), 403
    
    data = request.json
    alert = Alert(
        drug_name=data['drug_name'],
        message=data['message'],
        severity=data['severity'],
        sender_id=session['user_id']
    )
    
    db.session.add(alert)
    db.session.commit()
    
    return jsonify({'success': True, 'alert_id': alert.id})

@app.route('/api/drug-advisories', methods=['GET'])
def get_drug_advisories():
    drugs = Drug.query.order_by(Drug.created_at.desc()).limit(100).all()
    
    advisories = []
    for drug in drugs:
        # Map AI risk assessment to advisory type and severity
        advisory_type = 'advisory'
        severity = drug.ai_risk_assessment.lower() if drug.ai_risk_assessment else 'low'
        
        if severity == 'high':
            advisory_type = 'blackbox'
        elif severity == 'medium':
            advisory_type = 'contraindication'
        elif drug.ai_risk_details and 'monitor' in drug.ai_risk_details.lower():
            advisory_type = 'monitoring'
        
        # Create summary from risk details or use default
        summary = drug.ai_risk_details[:100] + '...' if drug.ai_risk_details and len(drug.ai_risk_details) > 100 else (drug.ai_risk_details or 'Review drug information for safety considerations')
        
        advisories.append({
            'id': drug.id,
            'type': advisory_type,
            'drug': drug.name,
            'summary': summary,
            'description': drug.ai_risk_details or drug.description or 'No detailed information available.',
            'severity': severity if severity in ['low', 'medium', 'high'] else 'medium',
            'source': drug.company.name if drug.company else 'Pharma Company',
            'date': drug.created_at.isoformat(),
            'reference': None
        })
    
    return jsonify({
        'success': True,
        'advisories': advisories
    })

@app.route('/api/side-effect-reports', methods=['GET'])
def get_side_effect_reports():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    doctor_id = session['user_id']
    reports = SideEffectReport.query.filter_by(doctor_id=doctor_id).order_by(SideEffectReport.created_at.desc()).all()
    
    return jsonify({
        'success': True,
        'reports': [{
            'id': r.id,
            'reportType': 'patient-linked' if r.patient_id else 'anonymised',
            'drugName': r.drug_name,
            'dateSubmitted': r.created_at.isoformat(),
            'status': 'Submitted',
            'patientId': r.patient_id
        } for r in reports]
    })



@app.route('/api/alerts/<int:alert_id>/read', methods=['POST'])
def mark_alert_read(alert_id):
    alert = Alert.query.get(alert_id)
    if alert:
        alert.is_read = True
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 404

# Analytics APIs
@app.route('/api/analytics/advanced')
def get_advanced_analytics():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    
    if user.role == 'pharma':
        company_drugs = [d.name for d in Drug.query.filter_by(company_id=user.id).all()]
        patients = Patient.query.filter(Patient.drug_name.in_(company_drugs)).all() if company_drugs else []
        alerts = Alert.query.filter(Alert.drug_name.in_(company_drugs)).all() if company_drugs else []
    else:
        patients = Patient.query.all()
        alerts = Alert.query.all()
    
    # Age distribution
    age_groups = {'0-18': 0, '19-40': 0, '41-60': 0, '60+': 0}
    for p in patients:
        if p.age <= 18:
            age_groups['0-18'] += 1
        elif p.age <= 40:
            age_groups['19-40'] += 1
        elif p.age <= 60:
            age_groups['41-60'] += 1
        else:
            age_groups['60+'] += 1
    
    # Gender-Risk analysis
    gender_risk = {
        'Male': {'Low': 0, 'Medium': 0, 'High': 0},
        'Female': {'Low': 0, 'Medium': 0, 'High': 0},
        'Other': {'Low': 0, 'Medium': 0, 'High': 0}
    }
    for p in patients:
        if p.gender in gender_risk and p.risk_level in ['Low', 'Medium', 'High']:
            gender_risk[p.gender][p.risk_level] += 1
    
    # Top drugs by reports
    drug_counts = {}
    for p in patients:
        drug_counts[p.drug_name] = drug_counts.get(p.drug_name, 0) + 1
    top_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Alert severity distribution
    alert_severity = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
    for a in alerts:
        if a.severity in alert_severity:
            alert_severity[a.severity] += 1
    
    # Drug risk distribution
    drug_risk = {
        'Low': len([p for p in patients if p.risk_level == 'Low']),
        'Medium': len([p for p in patients if p.risk_level == 'Medium']),
        'High': len([p for p in patients if p.risk_level == 'High'])
    }
    
    # Age-Risk average
    age_by_risk = {'Low': [], 'Medium': [], 'High': []}
    for p in patients:
        if p.risk_level in age_by_risk:
            age_by_risk[p.risk_level].append(p.age)
    
    age_risk_avg = {
        'Low': sum(age_by_risk['Low']) / len(age_by_risk['Low']) if age_by_risk['Low'] else 0,
        'Medium': sum(age_by_risk['Medium']) / len(age_by_risk['Medium']) if age_by_risk['Medium'] else 0,
        'High': sum(age_by_risk['High']) / len(age_by_risk['High']) if age_by_risk['High'] else 0
    }
    
    # Monthly trend (last 12 months)
    from datetime import datetime, timedelta
    monthly_counts = {}
    for p in patients:
        month_key = p.created_at.strftime('%b %Y')
        monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
    
    # Sort by date and get last 12 months
    monthly_trend = [{'month': k, 'count': v} for k, v in sorted(monthly_counts.items())[-12:]]
    
    return jsonify({
        'success': True,
        'totalPatients': len(patients),
        'totalDrugs': len(drug_counts),
        'totalAlerts': len(alerts),
        'ageDistribution': age_groups,
        'genderRiskAnalysis': gender_risk,
        'topDrugsByReports': [{'drug': name, 'count': count} for name, count in top_drugs],
        'alertSeverityDistribution': alert_severity,
        'drugRiskDistribution': drug_risk,
        'ageRiskAverage': age_risk_avg,
        'monthlyTrend': monthly_trend
    })

# Pharmacy-specific APIs
@app.route('/api/pharmacy/stats')
def get_pharmacy_stats():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    # Get reports created by this pharmacy
    my_reports = Patient.query.filter_by(created_by=user.id).all()
    today_reports = [p for p in my_reports if p.created_at.date() == datetime.utcnow().date()]
    
    # Severity distribution
    severity_dist = {
        'High': len([p for p in my_reports if p.risk_level == 'High']),
        'Medium': len([p for p in my_reports if p.risk_level == 'Medium']),
        'Low': len([p for p in my_reports if p.risk_level == 'Low'])
    }
    
    # Top drugs
    drug_counts = {}
    for p in my_reports:
        drug_counts[p.drug_name] = drug_counts.get(p.drug_name, 0) + 1
    top_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return jsonify({
        'success': True,
        'today': len(today_reports),
        'total': len(my_reports),
        'alerts': Alert.query.filter_by(is_read=False).count(),
        'dispensing': len(my_reports) * 15,  # Approximate
        'severity': {
            'low': severity_dist['Low'],
            'medium': severity_dist['Medium'],
            'high': severity_dist['High']
        },
        'topDrugs': [{'name': name, 'count': count} for name, count in top_drugs]
    })

@app.route('/api/pharmacy/reports')
def get_pharmacy_reports():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    limit = request.args.get('limit', type=int)
    reports = Patient.query.filter_by(created_by=user.id).order_by(Patient.created_at.desc())
    
    if limit:
        reports = reports.limit(limit)
    
    return jsonify({
        'success': True,
        'reports': [{
            'id': p.id,
            'date': p.created_at.isoformat(),
            'patientName': p.name,
            'drugName': p.drug_name,
            'reaction': p.symptoms,
            'severity': p.risk_level,
            'status': 'Submitted'
        } for p in reports.all()]
    })

@app.route('/api/pharmacy/reports/submit', methods=['POST'])
def submit_pharmacy_reports():
    """Submit pharmacy safety data reports"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    try:
        data = request.json
        report_type = data.get('report_type', 'anonymous')
        records = data.get('records', [])
        
        created_records = []
        created_patients = []  # Store patients for auto-send followup
        skipped_duplicates = []  # Track rejected duplicates
        linked_cases = []  # Track linked cases
        
        # Field mapping: Excel labels (from frontend) -> backend keys
        FIELD_MAP = {
            'Drug Name': 'drug_name',
            'Batch / Lot Number': 'batch_lot_number',
            'Dosage Form': 'dosage_form',
            'Date of Dispensing': 'date_of_dispensing',
            'Reaction Category': 'reaction_category',
            'Severity': 'severity',
            'Reaction Outcome': 'reaction_outcome',
            'Age Group': 'age_group',
            'Gender': 'gender',
            'Additional Notes': 'additional_notes',
            'Internal Case ID': 'internal_case_id',
            'Treating Hospital / Doctor Reference': 'treating_hospital_reference',
            'Treating Doctor Name': 'treating_doctor_name',
            'Total Dispensed': 'total_dispensed',
            'Total Reactions Reported': 'total_reactions_reported',
            'Mild Count': 'mild_count',
            'Moderate Count': 'moderate_count',
            'Severe Count': 'severe_count',
            'Reporting Period Start': 'reporting_period_start',
            'Reporting Period End': 'reporting_period_end',
            'Analysis Notes': 'analysis_notes',
            'Email': 'email',
            'Phone': 'phone',
        }
        
        def normalize_record(record):
            """Normalize record keys from Excel labels to backend keys"""
            normalized = {}
            for key, value in record.items():
                # Check if key is an Excel label and map it
                if key in FIELD_MAP:
                    normalized[FIELD_MAP[key]] = value
                else:
                    # Already a backend key or unknown - use as-is (lowercase)
                    normalized[key.lower().replace(' ', '_')] = value
            return normalized
        
        for record in records:
            # Normalize Excel labels to backend keys
            record = normalize_record(record)
            
            # Extract patient data
            name = 'Anonymous' if report_type == 'anonymous' else record.get('internal_case_id', 'Patient')
            email = record.get('email')
            phone = record.get('phone')
            drug_name = record.get('drug_name', 'Unknown')
            symptoms = record.get('reaction_category', '')
            gender = record.get('gender', 'not_specified')
            age_group = record.get('age_group', 'adult')
            
            # Map age group to approximate age
            age_map = {'pediatric': 8, 'adolescent': 16, 'adult': 35, 'elderly': 70, 'unknown': 35}
            age = age_map.get(age_group, 35)
            
            # === DUPLICATE CHECK ===
            duplicate_check = check_duplicate_patient(
                name=name,
                drug_name=drug_name,
                age=age,
                gender=gender,
                symptoms=symptoms,
                email=email,
                phone=phone
            )
            
            if duplicate_check['action'] == 'REJECT':
                # Skip this duplicate entry
                existing = duplicate_check['existing_case']
                skipped_duplicates.append({
                    'name': name,
                    'drug_name': drug_name,
                    'reason': duplicate_check['reason'],
                    'existing_id': existing.id
                })
                continue
            
            # Generate unique patient ID
            patient_id = f"PH-{random.randint(10000, 99999)}"
            while Patient.query.get(patient_id):
                patient_id = f"PH-{random.randint(10000, 99999)}"
            
            # Determine mode based on contact info
            mode = 'identity' if (email or phone) else 'anonymous'
            
            if duplicate_check['action'] == 'LINK':
                # Create but link to existing case
                existing = duplicate_check['existing_case']
                patient = Patient(
                    id=patient_id,
                    created_by=user.id,
                    name=name,
                    email=email,
                    phone=phone,
                    age=age,
                    gender=gender,
                    drug_name=drug_name,
                    symptoms=symptoms,
                    mode=mode,
                    risk_level=record.get('severity', 'mild').capitalize(),
                    case_status='Linked',
                    linked_case_id=existing.id,
                    match_score=duplicate_check['match_score'],
                    match_notes=f"Auto-linked: {duplicate_check['reason']}"
                )
                linked_cases.append({
                    'new_id': patient_id,
                    'linked_to': existing.id,
                    'match_score': duplicate_check['match_score']
                })
            else:
                # Map form fields to Patient model
                patient = Patient(
                    id=patient_id,
                    created_by=user.id,
                    name=name,
                    email=email,
                    phone=phone,
                    age=age,
                    gender=gender,
                    drug_name=drug_name,
                    symptoms=symptoms,
                    mode=mode,
                    risk_level=record.get('severity', 'mild').capitalize()
                )
            
            db.session.add(patient)
            created_records.append(patient_id)
            created_patients.append(patient)
        
        db.session.commit()
        
        # Score each new patient record
        scoring_results = []
        for patient in created_patients:
            try:
                evaluate_case(patient)
                score_case(patient)
                check_followup(patient)
                scoring_results.append({'patient_id': patient.id, 'scored': True})
            except Exception as score_err:
                scoring_results.append({'patient_id': patient.id, 'scored': False, 'error': str(score_err)})
        
        db.session.commit()
        
        # Auto-send follow-up to all patients with email or phone
        followup_results = []
        for patient in created_patients:
            if patient.email or patient.phone:
                result = auto_send_followup(patient)
                followup_results.append({
                    'patient_id': patient.id,
                    'email': patient.email,
                    'phone': patient.phone,
                    'channels_sent': result.get('channels_sent', 0),
                    'success': result.get('success', False)
                })
        
        response_data = {
            'success': True,
            'message': f'{len(created_records)} record(s) submitted successfully',
            'record_ids': created_records
        }
        
        if skipped_duplicates:
            response_data['skipped_duplicates'] = skipped_duplicates
            response_data['message'] += f', {len(skipped_duplicates)} duplicate(s) skipped'
        
        if linked_cases:
            response_data['linked_cases'] = linked_cases
            response_data['message'] += f', {len(linked_cases)} case(s) linked to existing'
        
        if followup_results:
            response_data['followup_emails'] = followup_results
        
        return jsonify(response_data)
    except Exception as e:
        db.session.rollback()
        print(f"Error submitting pharmacy reports: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/pharmacy/reports/compliance-score')
def get_pharmacy_compliance_score():
    """Calculate compliance score for pharmacy based on reporting activity"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    try:
        from datetime import datetime, timedelta
        
        # Get all reports by this pharmacy user
        total_reports = Patient.query.filter_by(created_by=user.id).count()
        
        # Reports this month
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_reports = Patient.query.filter(
            Patient.created_by == user.id,
            Patient.created_at >= month_start
        ).count()
        
        # Reports this week
        week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        weekly_reports = Patient.query.filter(
            Patient.created_by == user.id,
            Patient.created_at >= week_start
        ).count()
        
        # Calculate compliance score (0-100)
        # Base score starts at 50
        score = 50
        
        # +20 points for having at least 5 total reports
        if total_reports >= 5:
            score += 20
        elif total_reports >= 1:
            score += total_reports * 4  # 4 points per report up to 5
        
        # +15 points for monthly reporting activity (at least 2 reports/month)
        if monthly_reports >= 2:
            score += 15
        elif monthly_reports >= 1:
            score += 8
        
        # +15 points for weekly reporting activity (at least 1 report/week)
        if weekly_reports >= 1:
            score += 15
        
        # Cap at 100
        score = min(100, score)
        
        # Determine status
        if score >= 90:
            status = "Excellent"
        elif score >= 80:
            status = "Good"
        elif score >= 60:
            status = "Satisfactory"
        elif score >= 40:
            status = "Needs Improvement"
        else:
            status = "At Risk"
        
        return jsonify({
            'success': True,
            'compliance_score': score,
            'status': status,
            'details': {
                'total_reports': total_reports,
                'monthly_reports': monthly_reports,
                'weekly_reports': weekly_reports
            }
        })
    except Exception as e:
        print(f"Error calculating compliance score: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/pharmacy/report', methods=['POST'])
def submit_pharmacy_report():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    data = request.json
    
    # Extract patient data
    name = data['patientName']
    phone = data.get('phone', '')
    email = data.get('email')
    age = data['age']
    gender = data['gender']
    drug_name = data['drugName']
    symptoms = data['reaction']
    
    # === DUPLICATE CHECK ===
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
        # Exact duplicate - reject the entry
        existing = duplicate_check['existing_case']
        return jsonify({
            'success': False,
            'duplicate': True,
            'message': duplicate_check['reason'],
            'existing_patient': {
                'id': existing.id,
                'name': existing.name,
                'drug_name': existing.drug_name,
                'created_at': existing.created_at.isoformat() if existing.created_at else None
            },
            'match_score': duplicate_check['match_score']
        }), 409
    
    # Generate pharmacy report ID
    patient_id = f"PH-{random.randint(1000, 9999)}"
    while Patient.query.get(patient_id):
        patient_id = f"PH-{random.randint(1000, 9999)}"
    
    if duplicate_check['action'] == 'LINK':
        # Similar case - create but link to existing
        existing = duplicate_check['existing_case']
        patient = Patient(
            id=patient_id,
            created_by=user.id,
            name=name,
            phone=phone,
            email=email,
            age=age,
            gender=gender,
            drug_name=drug_name,
            symptoms=symptoms,
            risk_level=data['severity'],
            case_status='Linked',
            linked_case_id=existing.id,
            match_score=duplicate_check['match_score'],
            match_notes=f"Auto-linked: {duplicate_check['reason']}"
        )
    else:
        # No duplicate - create new
        patient = Patient(
            id=patient_id,
            created_by=user.id,
            name=name,
            phone=phone,
            email=email,
            age=age,
            gender=gender,
            drug_name=drug_name,
            symptoms=symptoms,
            risk_level=data['severity']
        )
    
    db.session.add(patient)
    db.session.commit()
    
    # Auto-send follow-up email if patient has email
    followup_result = None
    if patient.email:
        followup_result = auto_send_followup_email(patient)
    
    response_data = {
        'success': True,
        'report_id': patient.id,
        'case_status': patient.case_status,
        'linked_case_id': patient.linked_case_id,
        'duplicate_check': {
            'action': duplicate_check['action'],
            'match_score': duplicate_check['match_score'],
            'reason': duplicate_check['reason']
        }
    }
    if followup_result:
        response_data['followup_email'] = followup_result
    
    return jsonify(response_data)


@app.route('/api/pharmacy/settings', methods=['GET'])
def get_pharmacy_settings():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    # Import PharmacySettings here to avoid circular imports
    from models import PharmacySettings
    
    # Get or create settings
    settings = PharmacySettings.query.filter_by(pharmacy_id=user.id).first()
    if not settings:
        settings = PharmacySettings(pharmacy_id=user.id)
        db.session.add(settings)
        db.session.commit()
    
    return jsonify({
        'success': True,
        **settings.to_dict()
    })

@app.route('/api/pharmacy/settings/account', methods=['POST'])
def save_pharmacy_account_settings():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    from models import PharmacySettings
    
    data = request.json
    
    # Get or create settings
    settings = PharmacySettings.query.filter_by(pharmacy_id=user.id).first()
    if not settings:
        settings = PharmacySettings(pharmacy_id=user.id)
    
    # Update account fields
    settings.phone = data.get('phone', settings.phone)
    settings.address = data.get('address', settings.address)
    settings.license = data.get('license', settings.license)
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Account settings saved'})

@app.route('/api/pharmacy/settings/privacy', methods=['POST'])
def save_pharmacy_privacy_settings():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    from models import PharmacySettings
    
    data = request.json
    
    # Get or create settings
    settings = PharmacySettings.query.filter_by(pharmacy_id=user.id).first()
    if not settings:
        settings = PharmacySettings(pharmacy_id=user.id)
    
    # Update privacy fields
    settings.share_reports = data.get('shareReports', settings.share_reports)
    settings.share_dispensing = data.get('shareDispensing', settings.share_dispensing)
    settings.anonymize_data = data.get('anonymizeData', settings.anonymize_data)
    settings.retention_period = data.get('retentionPeriod', settings.retention_period)
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Privacy preferences saved'})

@app.route('/api/pharmacy/settings/notifications', methods=['POST'])
def save_pharmacy_notification_settings():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    from models import PharmacySettings
    
    data = request.json
    
    # Get or create settings
    settings = PharmacySettings.query.filter_by(pharmacy_id=user.id).first()
    if not settings:
        settings = PharmacySettings(pharmacy_id=user.id)
    
    # Update notification fields
    settings.alert_frequency = data.get('alertFrequency', settings.alert_frequency)
    settings.notify_email = data.get('notifyEmail', settings.notify_email)
    settings.notify_sms = data.get('notifySms', settings.notify_sms)
    settings.notify_dashboard = data.get('notifyDashboard', settings.notify_dashboard)
    settings.alert_recalls = data.get('alertRecalls', settings.alert_recalls)
    settings.alert_safety = data.get('alertSafety', settings.alert_safety)
    settings.alert_interactions = data.get('alertInteractions', settings.alert_interactions)
    settings.alert_dosage = data.get('alertDosage', settings.alert_dosage)
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Notification preferences saved'})

@app.route('/api/pharmacy/settings/compliance', methods=['POST'])
def save_pharmacy_compliance_settings():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    from models import PharmacySettings
    
    data = request.json
    
    # Get or create settings
    settings = PharmacySettings.query.filter_by(pharmacy_id=user.id).first()
    if not settings:
        settings = PharmacySettings(pharmacy_id=user.id)
    
    # Update compliance fields
    settings.reporting_authority = data.get('reportingAuthority', settings.reporting_authority)
    settings.reporting_threshold = data.get('reportingThreshold', settings.reporting_threshold)
    settings.compliance_officer = data.get('complianceOfficer', settings.compliance_officer)
    settings.auto_report = data.get('autoReport', settings.auto_report)
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Compliance settings saved'})

@app.route('/api/pharmacy/alerts', methods=['GET'])
def get_pharmacy_alerts():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        # Get all alerts for this pharmacy (recipient_type = 'all' or specific pharmacy)
        alerts = Alert.query.filter(
            Alert.recipient_type.in_(['all', 'pharmacy'])
        ).order_by(Alert.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'alerts': [alert.to_dict() for alert in alerts]
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching alerts: {str(e)}'
        }), 500

@app.route('/api/pharmacy/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_pharmacy_alert(alert_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        # Find the alert
        alert = Alert.query.get(alert_id)
        if not alert:
            return jsonify({'success': False, 'message': 'Alert not found'}), 404
        
        # Update alert status
        alert.status = 'acknowledged'
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = user.id
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Alert acknowledged',
            'status': alert.status,
            'acknowledged_at': alert.acknowledged_at.isoformat()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error acknowledging alert: {str(e)}'
        }), 500

# Case Matching APIs for Duplicate Detection
@app.route('/api/cases/match', methods=['POST'])
def match_cases():
    """
    Check for matching cases before adding a new case
    Returns similar cases with similarity scores and recommendations
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    data = request.json
    
    # Prepare new case data
    new_case = {
        'drug_name': data.get('drugName', ''),
        'symptoms': data.get('symptoms', ''),
        'age': data.get('age'),
        'gender': data.get('gender', ''),
    }
    
    # Get threshold from request or use default
    threshold = data.get('threshold', 0.75)
    
    # Get existing cases for this drug
    existing_patients = Patient.query.filter_by(
        drug_name=new_case['drug_name']
    ).filter(Patient.case_status != 'Discarded').all()
    
    # Run matching algorithm
    match_result = match_new_case(new_case, existing_patients, threshold=threshold)
    
    # Get action recommendation
    action_recommendation = should_accept_case(match_result, auto_discard_threshold=0.90)
    
    return jsonify({
        'success': True,
        'matches': match_result['matches'],
        'total_matches': match_result['total_matches'],
        'has_exact_match': match_result['has_exact_match'],
        'recommendation': match_result['recommendation'],
        'action': action_recommendation['action'],
        'reason': action_recommendation['reason'],
        'threshold': threshold
    })

@app.route('/api/cases/link', methods=['POST'])
def link_cases():
    """
    Link a new case to an existing case (mark as duplicate)
    """
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    data = request.json
    new_case_id = data.get('newCaseId')
    linked_case_id = data.get('linkedCaseId')
    match_score = data.get('matchScore')
    match_notes = data.get('notes', '')
    
    # Get the new case
    new_case = Patient.query.get(new_case_id)
    if not new_case:
        return jsonify({'success': False, 'message': 'Case not found'}), 404
    
    # Link to existing case
    new_case.linked_case_id = linked_case_id
    new_case.match_score = match_score
    new_case.case_status = 'Linked'
    new_case.match_notes = f'Linked to case {linked_case_id}. Score: {match_score}. {match_notes}'
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'Case {new_case_id} linked to {linked_case_id}',
        'case': {
            'id': new_case.id,
            'status': new_case.case_status,
            'linkedCaseId': new_case.linked_case_id,
            'matchScore': new_case.match_score
        }
    })

@app.route('/api/cases/discard', methods=['POST'])
def discard_case():
    """
    Mark a case as discarded (duplicate)
    """
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    data = request.json
    case_id = data.get('caseId')
    reason = data.get('reason', '')
    
    case = Patient.query.get(case_id)
    if not case:
        return jsonify({'success': False, 'message': 'Case not found'}), 404
    
    case.case_status = 'Discarded'
    case.match_notes = f'Discarded: {reason}'
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'Case {case_id} discarded',
        'case': {
            'id': case.id,
            'status': case.case_status
        }
    })

@app.route('/api/cases/<case_id>/details', methods=['GET'])
def get_case_details(case_id):
    """
    Get details of a specific case including linkage info
    """
    case = Patient.query.get(case_id)
    if not case:
        return jsonify({'success': False}), 404
    
    return jsonify({
        'success': True,
        'case': {
            'id': case.id,
            'name': case.name,
            'age': case.age,
            'gender': case.gender,
            'drugName': case.drug_name,
            'symptoms': case.symptoms,
            'riskLevel': case.risk_level,
            'status': case.case_status,
            'linkedCaseId': case.linked_case_id,
            'matchScore': case.match_score,
            'matchNotes': case.match_notes,
            'createdAt': case.created_at.isoformat(),
            'createdBy': case.created_by
        }
    })

# ===== QUALITY ASSESSMENT & SCORING APIS (STEPS 7-10) =====

@app.route('/api/cases/<case_id>/evaluate-strength', methods=['POST'])
def evaluate_case_strength(case_id):
    """
    STEP 7: Evaluate case quality and strength
    Returns completeness, temporal clarity, medical confirmation, and follow-up scores
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    case = Patient.query.get(case_id)
    if not case:
        return jsonify({'success': False, 'message': 'Case not found'}), 404
    
    engine = CaseScoringEngine()
    strength_info = engine.evaluate_case_strength(case)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'case_id': case_id,
        'strength': strength_info
    })


@app.route('/api/cases/<case_id>/calculate-score', methods=['POST'])
def calculate_case_score(case_id):
    """
    STEP 8: Calculate final case score
    Final Score = Polarity × Strength
    Score ranges from -2 (strong AE) to +2 (strong non-AE)
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    case = Patient.query.get(case_id)
    if not case:
        return jsonify({'success': False, 'message': 'Case not found'}), 404
    
    # Evaluate strength if not already done
    if case.strength_score is None:
        engine = CaseScoringEngine()
        engine.evaluate_case_strength(case)
    
    # Calculate final score
    engine = CaseScoringEngine()
    score_info = engine.calculate_final_score(case)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'case_id': case_id,
        'scoring': score_info
    })


@app.route('/api/cases/<case_id>/followup-status', methods=['GET'])
def get_followup_status(case_id):
    """
    STEP 9: Check if case requires follow-up
    Returns follow-up triggers and whether follow-up has been sent
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    case = Patient.query.get(case_id)
    if not case:
        return jsonify({'success': False, 'message': 'Case not found'}), 404
    
    # Calculate score if not done
    if case.case_score is None:
        engine = CaseScoringEngine()
        engine.evaluate_case_strength(case)
        engine.calculate_final_score(case)
    
    # Check follow-up triggers
    engine = CaseScoringEngine()
    followup_info = engine.check_followup_triggers(case)
    
    # Get existing followups
    manager = FollowUpManager()
    existing_followups = manager.get_followups_for_case(case_id)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'case_id': case_id,
        'followup': followup_info,
        'followup_history': existing_followups,
        'follow_up_required': case.follow_up_required,
        'follow_up_sent': case.follow_up_sent,
        'response_received': case.follow_up_response is not None
    })


@app.route('/api/cases/<case_id>/activate-agents', methods=['POST'])
def activate_quality_agents(case_id):
    """
    STEP 10: Activate AI agents to improve case quality
    Agents request targeted information from patients, doctors, hospitals
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    case = Patient.query.get(case_id)
    if not case:
        return jsonify({'success': False, 'message': 'Case not found'}), 404
    
    # Get follow-up triggers
    engine = CaseScoringEngine()
    followup_info = engine.check_followup_triggers(case)
    
    # Activate agents
    orchestrator = QualityAgentOrchestrator()
    agents_result = orchestrator.activate_agents(case, followup_info)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'case_id': case_id,
        'agents': agents_result
    })


@app.route('/api/agents/<agent_id>/questions', methods=['GET'])
def get_agent_questions(agent_id):
    """
    Get the targeted questions for a specific agent
    """
    agent = CaseAgent.query.get(agent_id)
    if not agent:
        return jsonify({'success': False, 'message': 'Agent not found'}), 404
    
    orchestrator = QualityAgentOrchestrator()
    message = orchestrator.get_agent_questions(agent_id)
    
    return jsonify({
        'success': True,
        'agent_id': agent_id,
        'agent_type': agent.agent_type,
        'role': agent.role,
        'questions': agent.target_questions,
        'message': message
    })


@app.route('/api/agents/<agent_id>/submit-response', methods=['POST'])
def submit_agent_response(agent_id):
    """
    Submit responses to an agent's questions
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    data = request.json
    responses = data.get('responses')
    
    orchestrator = QualityAgentOrchestrator()
    result = orchestrator.submit_agent_response(agent_id, responses)
    
    return jsonify(result)


@app.route('/api/cases/<case_id>/agents', methods=['GET'])
def get_case_agents(case_id):
    """
    Get all agents assigned to a case
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    case = Patient.query.get(case_id)
    if not case:
        return jsonify({'success': False, 'message': 'Case not found'}), 404
    
    orchestrator = QualityAgentOrchestrator()
    agents_info = orchestrator.get_active_agents_for_case(case_id)
    
    return jsonify({
        'success': True,
        'agents': agents_info
    })


@app.route('/api/dashboard/kpi', methods=['GET'])
def get_kpi_dashboard():
    """
    KPI Dashboard: Show case quality metrics
    - Total cases
    - Strong cases (score +2 or -2)
    - Medium cases (score +1 or -1)
    - Weak cases (score 0)
    - Distribution by strength
    - Recent trends
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    
    # Get relevant cases based on user role
    if user.role == 'pharma':
        company_drugs = [d.name for d in Drug.query.filter_by(company_id=user.id).all()]
        cases = Patient.query.filter(Patient.drug_name.in_(company_drugs)).all() if company_drugs else []
    elif user.role == 'doctor':
        cases = Patient.query.filter(Patient.doctors.contains(user)).all()
    else:
        cases = Patient.query.all()
    
    # Filter out linked/discarded cases
    active_cases = [c for c in cases if c.case_status == 'Active']
    
    # Calculate KPIs
    total_cases = len(active_cases)
    
    # Case strength distribution
    strong_cases = len([c for c in active_cases if c.strength_score == 2])
    medium_cases = len([c for c in active_cases if c.strength_score == 1])
    weak_cases = len([c for c in active_cases if c.strength_score == 0])
    not_evaluated = len([c for c in active_cases if c.strength_score is None])
    
    # Case score distribution
    strong_ae = len([c for c in active_cases if c.case_score == -2])
    weak_ae = len([c for c in active_cases if c.case_score == -1])
    unclear = len([c for c in active_cases if c.case_score == 0])
    weak_positive = len([c for c in active_cases if c.case_score == 1])
    strong_positive = len([c for c in active_cases if c.case_score == 2])
    
    # Follow-up required
    followup_required = len([c for c in active_cases if c.follow_up_required])
    followup_sent = len([c for c in active_cases if c.follow_up_sent])
    followup_received = len([c for c in active_cases if c.follow_up_response is not None])
    
    # Average scores
    avg_completeness = sum([c.completeness_score for c in active_cases if c.completeness_score]) / len([c for c in active_cases if c.completeness_score]) if [c for c in active_cases if c.completeness_score] else 0
    avg_temporal = sum([c.temporal_clarity_score for c in active_cases if c.temporal_clarity_score]) / len([c for c in active_cases if c.temporal_clarity_score]) if [c for c in active_cases if c.temporal_clarity_score] else 0
    avg_confirmation = sum([c.medical_confirmation_score for c in active_cases if c.medical_confirmation_score]) / len([c for c in active_cases if c.medical_confirmation_score]) if [c for c in active_cases if c.medical_confirmation_score] else 0
    
    return jsonify({
        'success': True,
        'overview': {
            'total_cases': total_cases,
            'cases_evaluated': total_cases - not_evaluated,
            'pending_evaluation': not_evaluated
        },
        'strength_distribution': {
            'strong': strong_cases,
            'medium': medium_cases,
            'weak': weak_cases,
            'not_evaluated': not_evaluated
        },
        'case_score_distribution': {
            'strong_ae': strong_ae,
            'weak_ae': weak_ae,
            'unclear': unclear,
            'weak_positive': weak_positive,
            'strong_positive': strong_positive
        },
        'followup_status': {
            'required': followup_required,
            'sent': followup_sent,
            'responses_received': followup_received,
            'pending_response': followup_sent - followup_received
        },
        'quality_metrics': {
            'avg_completeness': round(avg_completeness, 2),
            'avg_temporal_clarity': round(avg_temporal, 2),
            'avg_medical_confirmation': round(avg_confirmation, 2)
        },
        'trends': {
            'strong_vs_weak_ratio': f"{strong_cases}:{weak_cases}" if weak_cases > 0 else f"{strong_cases}:0",
            'quality_trend': 'Improving' if avg_completeness > 0.6 else 'Needs Improvement'
        }
    })


@app.route('/api/dashboard/case-details/<case_id>', methods=['GET'])
def get_case_quality_details(case_id):
    """
    Get detailed quality assessment for a specific case
    Shows all strength factors, score, agents, and follow-ups
    """
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    case = Patient.query.get(case_id)
    if not case:
        return jsonify({'success': False, 'message': 'Case not found'}), 404
    
    # Get agents
    agents = CaseAgent.query.filter_by(case_id=case_id).all()
    
    # Get followups
    followups = FollowUp.query.filter_by(case_id=case_id).all()
    
    return jsonify({
        'success': True,
        'case': {
            'id': case.id,
            'name': case.name,
            'drug_name': case.drug_name,
            'symptoms': case.symptoms,
            'risk_level': case.risk_level,
            'created_at': case.created_at.isoformat()
        },
        'strength_evaluation': {
            'strength_level': case.strength_level,
            'strength_score': case.strength_score,
            'completeness_score': round(case.completeness_score, 2) if case.completeness_score else 0,
            'temporal_clarity_score': round(case.temporal_clarity_score, 2) if case.temporal_clarity_score else 0,
            'medical_confirmation_score': round(case.medical_confirmation_score, 2) if case.medical_confirmation_score else 0,
            'followup_responsiveness_score': round(case.followup_responsiveness_score, 2) if case.followup_responsiveness_score else 0,
            'evaluated_at': case.evaluated_at.isoformat() if case.evaluated_at else None
        },
        'case_scoring': {
            'case_score': case.case_score,
            'polarity': case.polarity,
            'interpretation': case.case_score_interpretation,
            'calculated_at': case.case_score_calculated_at.isoformat() if case.case_score_calculated_at else None
        },
        'followup': {
            'required': case.follow_up_required,
            'sent': case.follow_up_sent,
            'response_received': case.follow_up_response is not None,
            'response': case.follow_up_response,
            'history': [
                {
                    'id': f.id,
                    'reason': f.reason,
                    'priority': f.priority,
                    'status': f.status,
                    'created_at': f.created_at.isoformat() if f.created_at else None,
                    'resolved_at': f.resolved_at.isoformat() if f.resolved_at else None
                } for f in followups
            ]
        },
        'agents': [
            {
                'id': a.id,
                'type': a.agent_type,
                'role': a.role,
                'status': a.status,
                'questions': a.target_questions,
                'responses': a.responses,
                'created_at': a.created_at.isoformat() if a.created_at else None,
                'resolved_at': a.resolved_at.isoformat() if a.resolved_at else None
            } for a in agents
        ]
    })

@app.route('/pharma/patient-recall')
def pharma_patient_recall():
    if 'user_id' not in session or session.get('role') != 'pharma':
        return redirect(url_for('login_page'))
    return render_template('pharma/patient-recall.html', active_page='patient-recall')

@app.route('/api/patients/recalled', methods=['GET'])
def get_recalled_patients():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
    try:
        # Get all recalled patients
        recalled_patients = Patient.query.filter_by(recalled=True).order_by(Patient.recall_date.desc()).all()
        return jsonify({
            'success': True,
            'patients': [p.to_dict() for p in recalled_patients]
        })
    except Exception as e:
        print(f"Error fetching recalled patients: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/patients/<patient_id>/recall', methods=['POST'])
def recall_patient(patient_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
    try:
        data = request.json
        reason = data.get('reason')
        
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({'success': False, 'message': 'Patient not found'}), 404
            
        patient.recalled = True
        patient.recall_reason = reason
        patient.recall_date = datetime.utcnow()
        patient.recalled_by = session['user_id']
        patient.follow_up_required = True  # Flag for hospital dashboard
        
        # Create a critical alert for the hospital dashboard
        alert = Alert(
            drug_name=patient.drug_name,
            title=f"URGENT: Patient Recall - {patient.name}",
            message=f"Patient {patient.name} (ID: {patient.id}) has been recalled. Reason: {reason}",
            severity='Critical',
            sender_id=session['user_id'],
            recipient_type='all', # Ideally target specific doctors/hospitals linked to patient
            status='new'
        )
        db.session.add(alert)
        
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Patient recalled successfully'})
    except Exception as e:
        db.session.rollback()
        print(f"Error recalling patient: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Hospital Routes
@app.route('/hospital/dashboard')
def hospital_dashboard():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return redirect(url_for('login_page'))
    return render_template('hospital/dashboad.html', active_page='overview')

@app.route('/hospital/patient-data')
def hospital_patient_data():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return redirect(url_for('login_page'))
    return render_template('hospital/patient-data.html', active_page='patient-data')

@app.route('/hospital/pharma-alerts')
def hospital_pharma_alerts():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return redirect(url_for('login_page'))
    return render_template('hospital/pharma-alerts.html', active_page='pharma-alerts')

@app.route('/hospital/patient-recall')
def hospital_patient_recall():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return redirect(url_for('login_page'))
    return render_template('hospital/patient-recall.html', active_page='patient-recall')

@app.route('/hospital/settings')
def hospital_settings():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return redirect(url_for('login_page'))
    return render_template('hospital/settings.html', active_page='settings')

@app.route('/hospital/doctors')
def hospital_doctors():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return redirect(url_for('login_page'))
    return render_template('hospital/doctors.html', active_page='doctors')

@app.route('/hospital/drugs')
def hospital_drugs():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return redirect(url_for('login_page'))
    return render_template('hospital/drugs.html', active_page='drugs')

# Hospital API Endpoints
@app.route('/api/hospital/info')
def get_hospital_info():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    return jsonify({
        'success': True,
        'hospitalName': session.get('hospital_name', 'General Hospital'),
        'hospitalId': session.get('user_id')
    })

@app.route('/api/hospital/drugs')
def get_hospital_drugs():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    # Common drugs list as fallback
    common_drugs = [
        'Aspirin', 'Ibuprofen', 'Paracetamol', 'Amoxicillin', 'Metformin',
        'Lisinopril', 'Atorvastatin', 'Omeprazole', 'Sertraline', 'Levothyroxine',
        'Amlodipine', 'Metoprolol', 'Ciprofloxacin', 'Azithromycin', 'Cephalexin',
        'Fluconazole', 'Loratadine', 'Cetirizine', 'Ranitidine', 'Diclofenac',
        'Naproxen', 'Tramadol', 'Codeine', 'Morphine', 'Insulin', 'Glucagon',
        'Warfarin', 'Heparin', 'Clopidogrel', 'Simvastatin', 'Pravastatin',
        'Rosuvastatin', 'Valsartan', 'Losartan', 'Enalapril', 'Ramipril',
        'Diltiazem', 'Verapamil', 'Nifedipine', 'Hydralazine', 'Furosemide',
        'Spironolactone', 'Hydrochlorothiazide', 'Albuterol', 'Salbutamol',
        'Fluticasone', 'Montelukast', 'Theophylline', 'Prednisone', 'Dexamethasone',
        'Methylprednisolone', 'PainAway', 'Neurocalm', 'Cardiostat', 'Immunoboost',
        'AntibioX'
    ]
    
    return jsonify({
        'success': True,
        'drugs': [{'name': drug} for drug in common_drugs]
    })

@app.route('/api/hospital/drug-stats/<drug_name>')
def get_hospital_drug_stats(drug_name):
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    # Generate demo statistics for the selected drug
    # In production, this would query actual hospital data
    import random
    
    total_patients = random.randint(50, 200)
    mildly_affected = random.randint(5, 30)
    adversely_affected = random.randint(2, 15)
    pharma_calls = random.randint(1, 10)
    
    return jsonify({
        'success': True,
        'drug': drug_name,
        'totalPatients': total_patients,
        'mildlyAffected': mildly_affected,
        'adverselyAffected': adversely_affected,
        'pharmaCalls': pharma_calls
    })

# Hospital Settings APIs
@app.route('/api/hospital/settings', methods=['POST'])
def save_hospital_settings():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    data = request.json
    user = User.query.get(session['user_id'])
    
    if user:
        # In a real app, you would save these to the database
        # For now, we'll just return success
        return jsonify({'success': True, 'message': 'Settings saved'})
    
    return jsonify({'success': False, 'message': 'User not found'}), 404

@app.route('/api/hospital/privacy-settings', methods=['POST'])
def save_privacy_settings():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    data = request.json
    user = User.query.get(session['user_id'])
    
    if user:
        # In a real app, you would save these to the database
        return jsonify({'success': True, 'message': 'Privacy settings saved'})
    
    return jsonify({'success': False, 'message': 'User not found'}), 404

@app.route('/api/hospital/notification-settings', methods=['POST'])
def save_notification_settings():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    data = request.json
    user = User.query.get(session['user_id'])
    
    if user:
        # In a real app, you would save these to the database
        return jsonify({'success': True, 'message': 'Notification settings saved'})
    
    return jsonify({'success': False, 'message': 'User not found'}), 404

# Doctor Settings APIs
@app.route('/api/doctor/settings', methods=['POST'])
def save_doctor_settings():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    data = request.json
    user = User.query.get(session['user_id'])
    
    if user:
        return jsonify({'success': True, 'message': 'Settings saved'})
    
    return jsonify({'success': False, 'message': 'User not found'}), 404

@app.route('/api/doctor/privacy-settings', methods=['POST'])
def save_doctor_privacy_settings():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    data = request.json
    user = User.query.get(session['user_id'])
    
    if user:
        return jsonify({'success': True, 'message': 'Privacy settings saved'})
    
    return jsonify({'success': False, 'message': 'User not found'}), 404

@app.route('/api/doctor/notification-settings', methods=['POST'])
def save_doctor_notification_settings():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    data = request.json
    user = User.query.get(session['user_id'])
    
    if user:
        return jsonify({'success': True, 'message': 'Notification settings saved'})
    
    return jsonify({'success': False, 'message': 'User not found'}), 404

# ========================================================================
# HOSPITAL MANAGEMENT APIs
# ========================================================================

@app.route('/api/hospital/patients', methods=['GET'])
def get_hospital_patients():
    """Get all patients linked to this hospital's doctors OR created by this hospital"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    try:
        hospital_id = session['user_id']
        patients_data = []
        seen_patient_ids = set()
        
        # First, get patients created directly by this hospital
        hospital_created_patients = Patient.query.filter_by(created_by=hospital_id).all()
        for patient in hospital_created_patients:
            if patient.id not in seen_patient_ids:
                seen_patient_ids.add(patient.id)
                patients_data.append({
                    'id': patient.id,
                    'name': patient.name,
                    'phone': patient.phone,
                    'email': patient.email,
                    'age': patient.age,
                    'gender': patient.gender,
                    'drugName': patient.drug_name,
                    'symptoms': patient.symptoms,
                    'riskLevel': patient.risk_level,
                    'created_at': patient.created_at.isoformat() if patient.created_at else None
                })
        
        # Also get patients from doctors linked to this hospital
        doctor_ids = db.session.query(hospital_doctor.c.doctor_id).filter(
            hospital_doctor.c.hospital_id == hospital_id
        ).all()
        doctor_ids = [d[0] for d in doctor_ids]
        
        if doctor_ids:
            for doctor_id in doctor_ids:
                doctor_patients = db.session.query(Patient).join(
                    doctor_patient, Patient.id == doctor_patient.c.patient_id
                ).filter(doctor_patient.c.doctor_id == doctor_id).all()
                
                for patient in doctor_patients:
                    if patient.id not in seen_patient_ids:
                        seen_patient_ids.add(patient.id)
                        patients_data.append({
                            'id': patient.id,
                            'name': patient.name,
                            'phone': patient.phone,
                            'email': patient.email,
                            'age': patient.age,
                            'gender': patient.gender,
                            'drugName': patient.drug_name,
                            'symptoms': patient.symptoms,
                            'riskLevel': patient.risk_level,
                            'created_at': patient.created_at.isoformat() if patient.created_at else None
                        })
        
        return jsonify({'success': True, 'patients': patients_data})
    except Exception as e:
        print(f"Error fetching hospital patients: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/hospital/patients', methods=['POST'])
def create_hospital_patient():
    """Create a new patient from hospital dashboard"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    try:
        data = request.get_json()
        mode = data.get('mode', 'identity')
        
        # Generate unique patient ID
        patient_id = data.get('patientId') or f"PT-{random.randint(10000, 99999)}"
        while Patient.query.get(patient_id):
            patient_id = f"PT-{random.randint(10000, 99999)}"
        
        # Extract patient data based on mode
        if mode == 'identity':
            name = data.get('name', 'Unknown')
            phone = data.get('phone')
            email = data.get('email')
            address = data.get('address', '')
            state = data.get('state', '')
            district = data.get('district', '')
            age = int(data.get('age', 0))
            gender = data.get('gender', 'Unknown')
            drug_name = data.get('drugName', 'Not Specified')
            symptoms = data.get('symptoms', data.get('adverseEvent', ''))
            risk_level = data.get('riskLevel', 'Low')
        else:  # anonymized mode
            name = 'Anonymized Patient'
            phone = None
            email = None
            address = ''
            state = ''
            district = ''
            age = int(data.get('age', 35))
            gender = data.get('gender', 'Unknown')
            drug_name = data.get('drugName', 'Not Specified')
            symptoms = data.get('symptoms', data.get('adverseEvent', ''))
            risk_level = data.get('riskLevel', 'Low')
        
        # Create the patient
        patient = Patient(
            id=patient_id,
            name=name,
            phone=phone,
            email=email,
            age=age,
            gender=gender,
            drug_name=drug_name,
            symptoms=symptoms,
            risk_level=risk_level,
            created_by=session['user_id'],
            created_at=datetime.utcnow()
        )
        
        db.session.add(patient)
        
        # Link patient to a hospital doctor if available
        hospital_id = session['user_id']
        doctor_ids = db.session.query(hospital_doctor.c.doctor_id).filter(
            hospital_doctor.c.hospital_id == hospital_id
        ).first()
        
        if doctor_ids:
            doctor = User.query.get(doctor_ids[0])
            if doctor:
                patient.doctors.append(doctor)
        
        db.session.commit()
        
        # Auto-start PV Agent follow-up cycle if patient has phone or email
        followup_result = None
        if patient.phone or patient.email:
            followup_result = auto_send_followup(patient)
            print(f"📱 PV Agent result for hospital patient {patient.id}: {followup_result}")
        
        response_data = {
            'success': True,
            'message': 'Patient created successfully',
            'patient': {
                'id': patient.id,
                'name': patient.name,
                'phone': patient.phone,
                'age': patient.age,
                'gender': patient.gender,
                'drugName': patient.drug_name,
                'symptoms': patient.symptoms,
                'riskLevel': patient.risk_level,
                'created_at': patient.created_at.isoformat()
            }
        }
        
        # Include follow-up result if sent
        if followup_result:
            response_data['followup'] = followup_result
        
        return jsonify(response_data)
        
    except Exception as e:
        db.session.rollback()
        print(f"Error creating hospital patient: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/hospital/doctors', methods=['GET'])
def get_hospital_doctors():
    """Get all doctors registered under this hospital"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    hospital = User.query.get(session['user_id'])
    if not hospital:
        return jsonify({'success': False, 'message': 'Hospital not found'}), 404
    
    # Get doctors linked to this hospital via hospital_doctor table
    # Query association table directly to avoid ambiguous foreign key error
    doctor_ids = db.session.query(hospital_doctor.c.doctor_id).filter(
        hospital_doctor.c.hospital_id == hospital.id
    ).all()
    doctor_ids = [did[0] for did in doctor_ids]
    
    doctors = User.query.filter(User.id.in_(doctor_ids), User.role == 'doctor').all()
    
    doctors_list = [{
        'id': doc.id,
        'name': doc.name,
        'email': doc.email,
        'specialization': doc.hospital_name  # Can be repurposed for specialization
    } for doc in doctors]
    
    return jsonify({'success': True, 'doctors': doctors_list})

@app.route('/api/hospital/drugs-in-use', methods=['GET'])
def get_hospital_drugs_in_use():
    """Get all drugs in use at this hospital"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    hospital = User.query.get(session['user_id'])
    if not hospital:
        return jsonify({'success': False, 'message': 'Hospital not found'}), 404
    
    # Get drugs linked to this hospital via hospital_drug table
    # Query association table directly to avoid ambiguous foreign key issues
    drug_ids = db.session.query(hospital_drug.c.drug_id).filter(
        hospital_drug.c.hospital_id == hospital.id
    ).all()
    drug_ids = [did[0] for did in drug_ids]
    
    drugs = Drug.query.filter(Drug.id.in_(drug_ids)).all()
    
    drugs_list = [drug.to_dict() for drug in drugs]
    
    return jsonify({'success': True, 'drugs': drugs_list})

@app.route('/api/hospital/pharmacies', methods=['GET'])
def get_hospital_pharmacies():
    """Get all pharmacies in contact with this hospital"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    hospital = User.query.get(session['user_id'])
    if not hospital:
        return jsonify({'success': False, 'message': 'Hospital not found'}), 404
    
    # Get pharmacies linked to this hospital via hospital_pharmacy table
    # Query association table directly to avoid ambiguous foreign key error
    pharmacy_ids = db.session.query(hospital_pharmacy.c.pharmacy_id).filter(
        hospital_pharmacy.c.hospital_id == hospital.id
    ).all()
    pharmacy_ids = [pid[0] for pid in pharmacy_ids]
    
    pharmacies = User.query.filter(User.id.in_(pharmacy_ids), User.role == 'pharmacy').all()
    
    pharmacies_list = [{
        'id': pharm.id,
        'name': pharm.name,
        'email': pharm.email
    } for pharm in pharmacies]
    
    return jsonify({'success': True, 'pharmacies': pharmacies_list})

@app.route('/api/hospital/side-effect-reports', methods=['GET'])
def get_hospital_side_effect_reports():
    """Get all side effect reports received by this hospital"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    reports = SideEffectReport.query.filter_by(hospital_id=session['user_id']).order_by(
        SideEffectReport.created_at.desc()
    ).all()
    
    reports_list = [report.to_dict() for report in reports]
    
    return jsonify({'success': True, 'reports': reports_list})

@app.route('/api/hospital/analytics', methods=['GET'])
def get_hospital_analytics():
    """Get comprehensive analytics for hospital dashboard"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    hospital_id = session['user_id']
    
    # Get hospital doctors by querying the association table
    doctor_ids = db.session.query(hospital_doctor.c.doctor_id).filter(
        hospital_doctor.c.hospital_id == hospital_id
    ).all()
    doctor_ids = [d[0] for d in doctor_ids]
    doctors = User.query.filter(User.id.in_(doctor_ids), User.role == 'doctor').all()
    
    # Get drugs in use by querying the association table
    drug_ids = db.session.query(hospital_drug.c.drug_id).filter(
        hospital_drug.c.hospital_id == hospital_id
    ).all()
    drug_ids = [d[0] for d in drug_ids]
    drugs = Drug.query.filter(Drug.id.in_(drug_ids)).all()
    
    # Get pharmacies by querying the association table
    pharmacy_ids = db.session.query(hospital_pharmacy.c.pharmacy_id).filter(
        hospital_pharmacy.c.hospital_id == hospital_id
    ).all()
    pharmacy_ids = [p[0] for p in pharmacy_ids]
    pharmacies = User.query.filter(User.id.in_(pharmacy_ids), User.role == 'pharmacy').all()
    
    # Get patients from hospital doctors
    patients = []
    for doctor in doctors:
        patients.extend(doctor.patients)
    
    # Get alerts sent to this hospital
    hospital_alerts = Alert.query.filter(
        (Alert.recipient_type == 'hospital') | (Alert.recipient_type == 'all')
    ).order_by(Alert.created_at.desc()).all()
    
    # Get side effect reports
    side_effect_reports = SideEffectReport.query.filter_by(hospital_id=hospital_id).all()
    
    # Calculate risk distribution in patients
    risk_distribution = {'High': 0, 'Medium': 0, 'Low': 0}
    for patient in patients:
        risk_level = patient.risk_level or 'Low'
        risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
    
    # Calculate drug companies distribution
    company_drug_count = {}
    for drug in drugs:
        company_name = drug.company.name if drug.company else 'Unknown'
        company_drug_count[company_name] = company_drug_count.get(company_name, 0) + 1
    
    # Calculate severity distribution in alerts
    severity_distribution = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
    for alert in hospital_alerts:
        severity = alert.severity or 'Low'
        severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
    
    # Doctor specialties distribution
    specialty_distribution = {}
    for doctor in doctors:
        # Extract specialty from name if present (e.g., "Dr. Emily Chen (Cardiology)")
        if '(' in doctor.name and ')' in doctor.name:
            specialty = doctor.name.split('(')[1].split(')')[0]
            specialty_distribution[specialty] = specialty_distribution.get(specialty, 0) + 1
    
    analytics = {
        'total_doctors': len(doctors),
        'total_drugs': len(drugs),
        'total_pharmacies': len(pharmacies),
        'total_patients': len(patients),
        'total_alerts': len(hospital_alerts),
        'total_side_effects': len(side_effect_reports),
        'risk_distribution': risk_distribution,
        'company_drug_count': company_drug_count,
        'severity_distribution': severity_distribution,
        'specialty_distribution': specialty_distribution,
        'recent_alerts': [alert.to_dict() for alert in hospital_alerts[:10]],
        'doctors_list': [{'id': d.id, 'name': d.name, 'email': d.email} for d in doctors[:10]],
        'top_drugs': [{'id': d.id, 'name': d.name, 'company': d.company.name if d.company else 'Unknown'} for d in drugs[:15]]
    }
    
    return jsonify({'success': True, 'analytics': analytics})

# ========================================================================
# SIDE EFFECT REPORTING APIs
# ========================================================================

@app.route('/api/report-side-effect', methods=['POST'])
def report_side_effect():
    """
    Doctor reports a side effect - with integrated case matching
    
    Flow:
    1. Check if report links to existing patient or needs new patient record
    2. Use case matching to find potential duplicates
    3. Create/link patient record as needed
    4. Create side effect report
    5. Send alerts to pharma company
    """
    if 'user_id' not in session or session.get('role') != 'doctor':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    try:
        data = request.json
        doctor = User.query.get(session['user_id'])
        report_type = data.get('report_type', 'anonymised')
        
        # Find the drug and its company
        drug = Drug.query.filter_by(name=data.get('drug_name')).first()
        
        patient_id = data.get('patient_id')
        case_matching_result = None
        new_patient_created = False
        
        # === STEP 1: Handle Patient Record ===
        if report_type == 'patient-linked':
            if patient_id:
                # Using existing patient from dropdown
                existing_patient = Patient.query.get(patient_id)
                if not existing_patient:
                    return jsonify({'success': False, 'message': f'Patient {patient_id} not found'}), 404
            else:
                # New patient - need to check case matching first
                patient_name = data.get('patient_name')
                patient_age = data.get('patient_age')
                patient_gender = data.get('patient_gender')
                symptoms = data.get('side_effect', '')
                
                # Validate required patient fields for new patient creation
                if not patient_name or not str(patient_name).strip():
                    return jsonify({'success': False, 'message': 'Patient name is required for patient-linked reports'}), 400
                if not patient_age:
                    return jsonify({'success': False, 'message': 'Patient age is required for patient-linked reports'}), 400
                if not patient_gender or not str(patient_gender).strip():
                    return jsonify({'success': False, 'message': 'Patient gender is required for patient-linked reports'}), 400
                
                # Prepare case data for matching
                new_case_data = {
                    'drug_name': data.get('drug_name', ''),
                    'symptoms': symptoms,
                    'age': int(patient_age),
                    'gender': patient_gender
                }
                
                # === STEP 2: Case Matching - Check for duplicates ===
                # Get existing patients for this drug
                existing_patients = Patient.query.filter_by(
                    drug_name=data.get('drug_name')
                ).filter(Patient.case_status != 'Discarded').all()
                
                match_result = match_new_case(new_case_data, existing_patients, threshold=0.75)
                action_recommendation = should_accept_case(match_result, auto_discard_threshold=0.90)
                
                case_matching_result = {
                    'action': action_recommendation['action'],
                    'reason': action_recommendation['reason'],
                    'total_matches': match_result.get('total_matches', 0),
                    'top_match': match_result.get('top_match', None)
                }
                
                # === STEP 3: Create or Link Patient Based on Match ===
                if action_recommendation['action'] == 'ACCEPT' or match_result.get('total_matches', 0) == 0:
                    # No match found - create new patient record
                    patient_id = f"PT-{random.randint(10000, 99999)}"
                    while Patient.query.get(patient_id):
                        patient_id = f"PT-{random.randint(10000, 99999)}"
                    
                    new_patient = Patient(
                        id=patient_id,
                        created_by=doctor.id,
                        name=patient_name,
                        phone=data.get('patient_phone'),
                        email=data.get('patient_email'),
                        age=int(patient_age),
                        gender=patient_gender,
                        drug_name=data.get('drug_name'),
                        symptoms=symptoms,
                        risk_level=data.get('severity', 'Medium'),
                        case_status='Active'
                    )
                    
                    # Add doctor-patient relationship
                    new_patient.doctors.append(doctor)
                    
                    db.session.add(new_patient)
                    db.session.flush()  # Get the ID without committing
                    
                    new_patient_created = True
                    case_matching_result['action'] = 'NEW'
                    case_matching_result['patient_id'] = patient_id
                    
                elif action_recommendation['action'] == 'REVIEW' or action_recommendation['action'] == 'DISCARD':
                    # Potential duplicate found - link to existing case
                    top_match = match_result['top_match']
                    linked_patient_id = top_match['case_id']
                    
                    # Create new patient record but link it to the existing case
                    patient_id = f"PT-{random.randint(10000, 99999)}"
                    while Patient.query.get(patient_id):
                        patient_id = f"PT-{random.randint(10000, 99999)}"
                    
                    new_patient = Patient(
                        id=patient_id,
                        created_by=doctor.id,
                        name=patient_name,
                        phone=data.get('patient_phone'),
                        email=data.get('patient_email'),
                        age=int(patient_age),
                        gender=patient_gender,
                        drug_name=data.get('drug_name'),
                        symptoms=symptoms,
                        risk_level=data.get('severity', 'Medium'),
                        case_status='Linked',
                        linked_case_id=linked_patient_id,
                        match_score=top_match['similarity_score'],
                        match_notes=f"Auto-linked: {action_recommendation['reason']}. Confidence: {top_match['confidence']}"
                    )
                    
                    new_patient.doctors.append(doctor)
                    db.session.add(new_patient)
                    db.session.flush()
                    
                    new_patient_created = True
                    case_matching_result['action'] = 'LINKED'
                    case_matching_result['patient_id'] = patient_id
                    case_matching_result['linked_to'] = linked_patient_id
                    case_matching_result['match_score'] = top_match['similarity_score']
        
        # === STEP 4: Create Side Effect Report ===
        report = SideEffectReport(
            patient_id=patient_id,
            doctor_id=doctor.id,
            hospital_id=None,
            drug_name=data.get('drug_name'),
            drug_id=drug.id if drug else None,
            side_effect=data.get('side_effect'),
            severity=data.get('severity', 'Medium'),
            company_notified=True if drug else False,
            hospital_notified=False
        )
        
        db.session.add(report)
        
        # === STEP 5: Case Scoring - Evaluate & Score the Case ===
        case_scoring_result = None
        if new_patient_created and patient_id:
            # Get the newly created patient
            created_patient = Patient.query.get(patient_id)
            if created_patient:
                # Mark as doctor confirmed since doctor is submitting
                created_patient.doctor_confirmed = True
                created_patient.doctor_confirmation_date = datetime.utcnow()
                
                # Evaluate case strength (Step 7)
                strength_result = evaluate_case(created_patient)
                
                # Calculate final score (Step 8)
                score_result = score_case(created_patient)
                
                # Check follow-up triggers (Step 9)
                followup_result = check_followup(created_patient)
                
                case_scoring_result = {
                    'strength': strength_result,
                    'score': score_result,
                    'followup': followup_result
                }
                
                # === STEP 5b: Auto-send Follow-up via Email AND Conversational WhatsApp ===
                if created_patient.email or created_patient.phone:
                    try:
                        from pv_backend.services.followup_agent import FollowupAgent
                        from pv_backend.routes.followup_routes import store_followup_token
                        from pv_backend.services.llm_service import PREDEFINED_QUESTIONS, get_combined_questions
                        from models import AgentFollowupTracking
                        
                        agent = FollowupAgent()
                        token = agent.generate_followup_token(created_patient.id)
                        store_followup_token(created_patient.id, token)
                        
                        channels_sent = 0
                        
                        # Get combined questions for the patient
                        questions_data = get_combined_questions(created_patient)
                        
                        # Send Email if patient has email (with form link)
                        if created_patient.email:
                            email_result = agent.send_followup_email(created_patient, token)
                            if email_result.get('success'):
                                channels_sent += 1
                                case_scoring_result['followup_email_sent'] = True
                                case_scoring_result['followup_email_to'] = created_patient.email
                            else:
                                case_scoring_result['followup_email_sent'] = False
                                case_scoring_result['followup_email_error'] = email_result.get('error')
                        
                        # Send Conversational WhatsApp if patient has phone number
                        if created_patient.phone:
                            # Create tracking record for conversational flow
                            tracking = AgentFollowupTracking.query.filter_by(
                                patient_id=created_patient.id,
                                status='active'
                            ).first()
                            
                            if not tracking:
                                tracking = AgentFollowupTracking(
                                    patient_id=created_patient.id,
                                    current_day=1,
                                    next_followup_date=datetime.utcnow(),
                                    predefined_questions=questions_data.get('predefined_questions', []),
                                    llm_questions=questions_data.get('llm_questions', []),
                                    day1_case_score=created_patient.case_score,
                                    chatbot_state='awaiting_language',  # Start conversational flow
                                    status='active'
                                )
                                db.session.add(tracking)
                                db.session.flush()
                            
                            # Send conversational WhatsApp (asks questions one by one)
                            whatsapp_result = agent.send_conversational_whatsapp(created_patient, tracking)
                            if whatsapp_result.get('success'):
                                channels_sent += 1
                                case_scoring_result['followup_whatsapp_sent'] = True
                                case_scoring_result['followup_whatsapp_to'] = created_patient.phone
                                case_scoring_result['followup_whatsapp_sid'] = whatsapp_result.get('message_sid')
                                case_scoring_result['followup_whatsapp_conversational'] = True
                                print(f"[OK] Conversational WhatsApp started with {created_patient.phone} - SID: {whatsapp_result.get('message_sid')}")
                            else:
                                case_scoring_result['followup_whatsapp_sent'] = False
                                case_scoring_result['followup_whatsapp_error'] = whatsapp_result.get('error')
                                print(f"[ERROR] WhatsApp failed for {created_patient.phone}: {whatsapp_result.get('error')}")
                        
                        # Update patient record if any channel succeeded
                        if channels_sent > 0:
                            created_patient.followup_sent_date = datetime.utcnow()
                            created_patient.followup_pending = True
                            created_patient.follow_up_sent = True
                            case_scoring_result['followup_channels_sent'] = channels_sent
                            
                    except Exception as followup_err:
                        print(f"Failed to send follow-up: {followup_err}")
                        import traceback
                        traceback.print_exc()
                        case_scoring_result['followup_email_sent'] = False
                        case_scoring_result['followup_whatsapp_sent'] = False
                        case_scoring_result['followup_error'] = str(followup_err)
        
        # === STEP 6: Send Alert to Pharma Company ===
        if drug and drug.company:
            severity_level = data.get('severity', 'Medium')
            alert_message = f"Side Effect Report: {data.get('drug_name')} - {data.get('side_effect')[:100]}"
            
            company_alert = Alert(
                drug_name=data.get('drug_name'),
                message=alert_message,
                severity=severity_level,
                sender_id=doctor.id,
                recipient_id=drug.company_id,
                recipient_type='pharma',
                is_read=False
            )
            db.session.add(company_alert)
        
        db.session.commit()
        
        response_data = {
            'success': True, 
            'message': 'Side effect reported successfully',
            'report_id': report.id,
            'patient_id': patient_id,
            'new_patient_created': new_patient_created,
            'hospital_notified': report.hospital_notified,
            'company_notified': report.company_notified
        }
        
        if case_matching_result:
            response_data['case_matching'] = case_matching_result
        
        if case_scoring_result:
            response_data['case_scoring'] = case_scoring_result
        
        return jsonify(response_data)
        
    except Exception as e:
        db.session.rollback()
        print(f"Error in report_side_effect: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


# ========================================================================
# AUTOMATIC DATABASE MIGRATION & POPULATION
# ========================================================================

def migrate_database():
    """
    Automatically add missing columns to database tables.
    This ensures the database schema matches the models without manual steps.
    """
    import sqlite3
    db_path = os.path.join(app.instance_path, 'inteleyzer.db')
    
    if not os.path.exists(db_path):
        return  # Database doesn't exist yet, will be created by db.create_all()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get existing columns for alert table
    cursor.execute('PRAGMA table_info(alert)')
    alert_cols = [col[1] for col in cursor.fetchall()]
    
    # Add missing columns to alert table
    alert_columns_to_add = [
        ('status', 'VARCHAR(20)', "'new'"),
        ('acknowledged_at', 'DATETIME', None),
        ('acknowledged_by', 'INTEGER', None)
    ]
    
    for col_name, col_type, default in alert_columns_to_add:
        if col_name not in alert_cols:
            try:
                if default:
                    sql = f'ALTER TABLE alert ADD COLUMN {col_name} {col_type} DEFAULT {default}'
                else:
                    sql = f'ALTER TABLE alert ADD COLUMN {col_name} {col_type}'
                cursor.execute(sql)
                print(f'[OK] Migration: Added {col_name} to alert table')
            except Exception as e:
                pass  # Column might already exist
    
    # Get existing columns for patient table
    cursor.execute('PRAGMA table_info(patient)')
    patient_cols = [col[1] for col in cursor.fetchall()]
    
    # Check if patient has all required columns
    patient_columns_to_add = [
        ('email', 'VARCHAR(120)', None),
        ('followup_sent_date', 'DATETIME', None),
        ('followup_pending', 'BOOLEAN', '0'),
        ('followup_completed', 'BOOLEAN', '0'),
        ('followup_response_date', 'DATETIME', None),
        ('followup_responded', 'BOOLEAN', '0')
    ]
    
    for col_name, col_type, default in patient_columns_to_add:
        if col_name not in patient_cols:
            try:
                if default:
                    sql = f'ALTER TABLE patient ADD COLUMN {col_name} {col_type} DEFAULT {default}'
                else:
                    sql = f'ALTER TABLE patient ADD COLUMN {col_name} {col_type}'
                cursor.execute(sql)
                print(f'[OK] Migration: Added {col_name} to patient table')
            except Exception as e:
                pass  # Column might already exist
    
    conn.commit()
    conn.close()

# Run migration before anything else
migrate_database()

# Automatically populate database on first run or if empty
if os.environ.get('SKIP_AUTO_POPULATE') != '1':
    with app.app_context():
        # Create all database tables
        db.create_all()
        
        # Check if database is empty
        from models import User
        user_count = User.query.count()
    
    if user_count == 0:
        print("\n" + "="*80)
        print("DATABASE IS EMPTY - STARTING AUTOMATIC POPULATION")
        print("="*80)
        from populate_enhanced_data import populate_database
        populate_database(app, db)
        print("\n" + "="*80)
        print("DATABASE POPULATION COMPLETE - Starting Flask server...")
        print("="*80 + "\n")
    else:
        print(f"\nDatabase already populated ({user_count} users found)")
else:
    # Skip auto-population
    with app.app_context():
        db.create_all()

# Initialize follow-up routes
init_followup_routes(app, db, Patient)

# ========================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
