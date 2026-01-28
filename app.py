"""
Inteleyzer - Pharmacovigilance Platform
Main Flask Application Entry Point
Supports: Pharmaceutical Companies, Doctors, and Local Pharmacies
Run on: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from models import db, User, Patient, Drug, Alert, SideEffectReport, hospital_doctor, hospital_drug, hospital_pharmacy, doctor_patient
from pv_backend.services.case_matching import match_new_case, should_accept_case
import os
import random
from datetime import datetime

app = Flask(__name__)
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

@app.route('/pharma/patient-recall')
def pharma_patient_recall():
    return render_template('pharma/patient-recall.html')

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
        session.permanent = True
        session['user_id'] = user.id
        session['role'] = user.role
        session['user_name'] = user.name
        # For hospital role, set hospital_name from user name
        if user.role == 'hospital':
            session['hospital_name'] = user.name
        return jsonify({
            'success': True, 
            'user': {'id': user.id, 'name': user.name, 'role': user.role}
        })
    
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/api/auth/logout', methods=['POST'])
def logout_api():
    session.clear()
    return jsonify({'success': True})

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
        'createdAt': p.created_at.isoformat()
    } for p in patients])

@app.route('/api/patients', methods=['POST'])
def create_patient():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'doctor':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        data = request.json
        
        # Simple risk logic
        risk = 'Low'
        symptoms = data.get('symptoms', '').lower()
        if len(symptoms) > 20:
            risk = 'Medium'
        if 'heart' in symptoms or 'breath' in symptoms or 'severe' in symptoms:
            risk = 'High'
        
        # Check if patient exists by phone
        existing_patient = Patient.query.filter_by(phone=data.get('phone')).first()
        
        if existing_patient:
            patient = existing_patient
            # Update fields
            patient.drug_name = data['drugName']
            if data.get('symptoms'):
                patient.symptoms = data.get('symptoms')
            patient.risk_level = risk
            patient.age = int(data['age'])
            # Ensure link to this doctor
            if user not in patient.doctors:
                patient.doctors.append(user)
        else:
            patient = Patient(
                id='PT-' + str(random.randint(1000, 9999)),
                created_by=session['user_id'],
                name=data['name'],
                phone=data.get('phone'),
                age=int(data['age']),
                gender=data['gender'],
                drug_name=data['drugName'],
                symptoms=data.get('symptoms'),
                risk_level=risk
            )
            # Link to doctor
            patient.doctors.append(user)
            db.session.add(patient)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'patient': {
                'id': patient.id,
                'name': patient.name,
                'age': patient.age,
                'gender': patient.gender,
                'drugName': patient.drug_name,
                'symptoms': patient.symptoms,
                'riskLevel': patient.risk_level,
                'phone': patient.phone,
                'createdAt': patient.created_at.isoformat()
            }
        })
    except Exception as e:
        db.session.rollback()
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
            'age': patient.age,
            'gender': patient.gender,
            'drug_name': patient.drug_name,
            'symptoms': patient.symptoms,
            'risk_level': patient.risk_level,
            'created_at': patient.created_at.isoformat(),
            'created_by': patient.created_by
        }
    })

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
        
    elif user.role == 'doctor':
        patients = Patient.query.filter(Patient.doctors.contains(user)).all()
        total_reports = len(patients)
        high_risk = len([p for p in patients if p.risk_level == 'High'])
        risk_dist = {'low': 0, 'medium': 0, 'high': 0}
        gender_dist = {'male': 0, 'female': 0, 'other': 0}
        avg_age = 0
    else:
        total_reports = 0
        high_risk = 0
        risk_dist = {'low': 0, 'medium': 0, 'high': 0}
        gender_dist = {'male': 0, 'female': 0, 'other': 0}
        avg_age = 0
    
    return jsonify({
        'success': True,
        'totalReports': total_reports,
        'highRiskCount': high_risk,
        'avgAge': round(avg_age, 1),
        'riskDist': risk_dist,
        'genderDist': gender_dist
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

# Patient Recall APIs
@app.route('/api/patients/<patient_id>/recall', methods=['POST'])
def recall_patient(patient_id):
    """Recall a patient for testing due to adverse effects"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharma':
        return jsonify({'success': False, 'message': 'Only pharmaceutical companies can recall patients'}), 403
    
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404
    
    # Verify this patient is taking a drug from this company
    company_drugs = [d.name for d in Drug.query.filter_by(company_id=user.id).all()]
    if patient.drug_name not in company_drugs:
        return jsonify({'success': False, 'message': 'Patient is not taking your company\'s drugs'}), 403
    
    try:
        data = request.json
        patient.recalled = True
        patient.recalled_by = user.id
        patient.recall_reason = data.get('reason', 'Further testing required')
        patient.recall_date = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'patient': patient.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/patients/recalled', methods=['GET'])
def get_recalled_patients():
    """Get all patients recalled by the pharma company"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharma':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    # Get all patients recalled by this company
    recalled = Patient.query.filter_by(recalled=True, recalled_by=user.id).order_by(Patient.recall_date.desc()).all()
    
    return jsonify({
        'success': True,
        'patients': [p.to_dict() for p in recalled]
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
    """Enhanced alert creation - broadcasts to ALL doctors and hospitals"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharma':
        return jsonify({'success': False, 'message': 'Only pharma companies can send alerts'}), 403
    
    try:
        data = request.json
        # Support both camelCase and snake_case for compatibility
        drug_name = data.get('drugName') or data.get('drug_name')
        
        if not drug_name:
            return jsonify({'success': False, 'message': 'Drug name is required'}), 400
        
        # Create the alert
        alert = Alert(
            drug_name=drug_name,
            title=data.get('title', f'{drug_name} Safety Alert'),
            message=data['message'],
            severity=data['severity'],
            sender_id=session['user_id'],
            recipient_type='all'  # Broadcasts to everyone
        )
        
        db.session.add(alert)
        db.session.commit()
        
        # Count recipients
        doctors_count = User.query.filter_by(role='doctor').count()
        hospitals_count = User.query.filter_by(role='hospital').count()
        
        return jsonify({
            'success': True, 
            'alert_id': alert.id, 
            'message': 'Alert sent successfully',
            'recipients': {
                'doctors': doctors_count,
                'hospitals': hospitals_count,
                'total': doctors_count + hospitals_count
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

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

@app.route('/api/pharmacy/report', methods=['POST'])
def submit_pharmacy_report():
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    data = request.json
    
    # Generate pharmacy report ID
    patient_id = f"PH-{random.randint(1000, 9999)}"
    while Patient.query.get(patient_id):
        patient_id = f"PH-{random.randint(1000, 9999)}"
    
    patient = Patient(
        id=patient_id,
        created_by=user.id,
        name=data['patientName'],
        phone=data.get('phone', ''),
        age=data['age'],
        gender=data['gender'],
        drug_name=data['drugName'],
        symptoms=data['reaction'],
        risk_level=data['severity']
    )
    
    db.session.add(patient)
    db.session.commit()
    
    return jsonify({'success': True, 'report_id': patient.id})

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

@app.route('/hospital/settings')
def hospital_settings():
    if 'user_id' not in session or session.get('role') != 'hospital':
        return redirect(url_for('login_page'))
    return render_template('hospital/settings.html', active_page='settings')

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

@app.route('/api/hospital/patients', methods=['GET'])
def get_hospital_patients():
    """Get all patients treated by doctors at this hospital"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    hospital = User.query.get(session['user_id'])
    if not hospital:
        return jsonify({'success': False, 'message': 'Hospital not found'}), 404
    
    # Get all doctors at this hospital
    doctor_ids = db.session.query(hospital_doctor.c.doctor_id).filter(
        hospital_doctor.c.hospital_id == hospital.id
    ).all()
    doctor_ids = [did[0] for did in doctor_ids]
    
    # Get all patients treated by these doctors
    patients_set = set()
    for doctor_id in doctor_ids:
        patient_ids = db.session.query(doctor_patient.c.patient_id).filter(
            doctor_patient.c.doctor_id == doctor_id
        ).all()
        patients_set.update([pid[0] for pid in patient_ids])
    
    patients = Patient.query.filter(Patient.id.in_(patients_set)).all() if patients_set else []
    
    return jsonify({
        'success': True,
        'patients': [p.to_dict() for p in patients]
    })

@app.route('/api/hospital/recalled-patients', methods=['GET'])
def get_hospital_recalled_patients():
    """Get all patients that have been recalled by pharma companies"""
    if 'user_id' not in session or session.get('role') != 'hospital':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    hospital = User.query.get(session['user_id'])
    if not hospital:
        return jsonify({'success': False, 'message': 'Hospital not found'}), 404
    
    # Get all doctors at this hospital
    doctor_ids = db.session.query(hospital_doctor.c.doctor_id).filter(
        hospital_doctor.c.hospital_id == hospital.id
    ).all()
    doctor_ids = [did[0] for did in doctor_ids]
    
    # Get recalled patients treated by these doctors
    patients_set = set()
    for doctor_id in doctor_ids:
        patient_ids = db.session.query(doctor_patient.c.patient_id).filter(
            doctor_patient.c.doctor_id == doctor_id
        ).all()
        patients_set.update([pid[0] for pid in patient_ids])
    
    recalled_patients = Patient.query.filter(
        Patient.id.in_(patients_set),
        Patient.recalled == True
    ).all() if patients_set else []
    
    return jsonify({
        'success': True,
        'patients': [p.to_dict() for p in recalled_patients]
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
    """Doctor reports a side effect - sends to hospital AND pharma company"""
    if 'user_id' not in session or session.get('role') != 'doctor':
        return jsonify({'success': False, 'message': 'Not authorized'}), 403
    
    data = request.json
    doctor = User.query.get(session['user_id'])
    
    # Check if doctor is registered under a hospital
    hospital = db.session.query(User).join(hospital_doctor).filter(
        hospital_doctor.c.doctor_id == doctor.id,
        User.role == 'hospital'
    ).first()
    
    # Find the drug and its company
    drug = Drug.query.filter_by(name=data.get('drug_name')).first()
    
    # Create side effect report
    report = SideEffectReport(
        patient_id=data.get('patient_id'),
        doctor_id=doctor.id,
        hospital_id=hospital.id if hospital else None,
        drug_name=data.get('drug_name'),
        drug_id=drug.id if drug else None,
        side_effect=data.get('side_effect'),
        severity=data.get('severity', 'Medium'),
        company_notified=True if drug else False,
        hospital_notified=True if hospital else False
    )
    
    db.session.add(report)
    db.session.commit()
    
    return jsonify({
        'success': True, 
        'message': 'Side effect reported successfully',
        'report_id': report.id,
        'hospital_notified': report.hospital_notified,
        'company_notified': report.company_notified
    })

if __name__ == '__main__':
    # ========================================================================
    # AUTOMATIC DATABASE POPULATION
    # ========================================================================
    # Automatically populate database on first run or if empty
    with app.app_context():
        # Check if database is empty
        from models import User
        user_count = User.query.count()
        
        if user_count == 0:
            print("\n" + "="*80)
            print("DATABASE IS EMPTY - STARTING AUTOMATIC POPULATION")
            print("="*80)
            from populate_enhanced_data import populate_database
            populate_database()
            print("\n" + "="*80)
            print("DATABASE POPULATION COMPLETE - Starting Flask server...")
            print("="*80 + "\n")
        else:
            print(f"\n✓ Database already populated ({user_count} users found)")
    
    # ========================================================================
    
    app.run(debug=True, host='127.0.0.1', port=5000)
