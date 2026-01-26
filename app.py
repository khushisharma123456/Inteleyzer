"""
MedSafe - Pharmacovigilance Platform
Main Flask Application Entry Point
Supports: Pharmaceutical Companies, Doctors, and Local Pharmacies
Run on: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from models import db, User, Patient, Drug, Alert
import os
import random
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medsafe-secret-key-dev'

# Get absolute path for database
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
os.makedirs(instance_path, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "medsafe.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db.init_app(app)

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
        session['user_id'] = user.id
        session['role'] = user.role
        session['user_name'] = user.name
        return jsonify({
            'success': True, 
            'user': {'id': user.id, 'name': user.name, 'role': user.role}
        })
    
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

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

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
