from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from models import db, User, Patient, Drug, Alert
import os
import random
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medsafe-secret-key-dev'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medsafe.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db.init_app(app)

# Create tables
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

@app.route('/doctor/dashboard')
def doctor_dashboard():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login_page'))
    return render_template('doctor/dashboard.html')

@app.route('/doctor/patients')
def doctor_patients():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login_page'))
    return render_template('doctor/patients_v2.html')

@app.route('/doctor/alerts')
def doctor_alerts():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login_page'))
    return render_template('doctor/alerts.html')

@app.route('/doctor/warnings')
def doctor_warnings():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login_page'))
    return render_template('doctor/warnings.html')

@app.route('/pharma/dashboard')
def pharma_dashboard():
    if 'user_id' not in session or session.get('role') != 'pharma':
        return redirect(url_for('login_page'))
    return render_template('pharma/dashboard.html')

@app.route('/pharma/reports')
def pharma_reports():
    if 'user_id' not in session or session.get('role') != 'pharma':
        return redirect(url_for('login_page'))
    return render_template('pharma/reports.html')

@app.route('/pharma/drugs')
def pharma_drugs():
    if 'user_id' not in session or session.get('role') != 'pharma':
        return redirect(url_for('login_page'))
    return render_template('pharma/drugs.html')

# --- API Endpoints ---

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'success': False, 'message': 'Email already exists'})
    
    new_user = User(
        name=data['name'],
        email=data['email'],
        password=data['password'], # Hash in prod
        role=data['role']
    )
    db.session.add(new_user)
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

@app.route('/api/patients', methods=['GET'])
def get_patients():
    try:
        role = session.get('role')
        user_id = session.get('user_id')
        
        if role == 'pharma':
            patients = Patient.query.all()
        else:
            # doctors = Patient.query.filter_by(doctor_id=user_id).all() # Old way
            user = User.query.get(user_id)
            patients = user.patients
            
        print(f"DEBUG: Found {len(patients)} patients")
        return jsonify([p.to_dict() for p in patients])
    except Exception as e:
        print(f"ERROR getting patients: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patients', methods=['POST'])
def create_patient():
    if session.get('role') != 'doctor':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        data = request.json
        
        # Simple risk logic
        risk = 'Low'
        symptoms = data.get('symptoms', '').lower()
        if len(symptoms) > 20: risk = 'Medium'
        if 'heart' in symptoms or 'breath' in symptoms or 'severe' in symptoms: risk = 'High'
        
        # Check if patient exists by phone
        existing_patient = Patient.query.filter_by(phone=data.get('phone')).first()
        current_doctor = User.query.get(session['user_id'])
        
        if existing_patient:
            print(f"DEBUG: Found existing patient {existing_patient.name} ({existing_patient.phone})")
            patient = existing_patient
            # Update fields
            patient.drug_name = data['drugName']
            if data.get('symptoms'):
                patient.symptoms = data.get('symptoms')
            patient.risk_level = risk
            patient.age = int(data['age'])
            # Ensure link to this doctor
            if current_doctor not in patient.doctors:
                patient.doctors.append(current_doctor)
        else:
            print("DEBUG: Creating new patient")
            patient = Patient(
                id='PT-' + str(random.randint(1000, 9999)),
                created_by=session['user_id'], # Track creator
                name=data['name'],
                phone=data.get('phone'),
                age=int(data['age']),
                gender=data['gender'],
                drug_name=data['drugName'],
                symptoms=data.get('symptoms'),
                risk_level=risk
            )
            # Link to doctor
            patient.doctors.append(current_doctor)
            db.session.add(patient)
        
        db.session.commit()
        return jsonify({'success': True, 'patient': patient.to_dict()})
    except Exception as e:
        print(f"ERROR creating patient: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/patients/<patient_id>/symptoms', methods=['POST'])
def update_symptoms(patient_id):
    try:
        data = request.json
        new_symptom = data.get('symptom')
        patient = Patient.query.get(patient_id)
        
        if not patient:
            return jsonify({'success': False, 'message': 'Patient not found'}), 404
            
        # Append new symptom with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d")
        current_symptoms = patient.symptoms or ""
        # Avoid double newline if empty
        if current_symptoms:
            updated_symptoms = f"{current_symptoms}\n[{timestamp}] {new_symptom}"
        else:
            updated_symptoms = f"[{timestamp}] {new_symptom}"
            
        patient.symptoms = updated_symptoms
        
        # Simple Risk Re-evaluation
        s_lower = updated_symptoms.lower()
        if 'heart' in s_lower or 'chest' in s_lower or 'severe' in s_lower or 'breath' in s_lower:
            patient.risk_level = 'High'
        elif len(s_lower) > 50 or 'dizzy' in s_lower:
            patient.risk_level = 'Medium'
        else:
            patient.risk_level = 'Low'
            
        db.session.commit()
        return jsonify({'success': True, 'patient': patient.to_dict()})
    except Exception as e:
        print(f"ERROR updating symptoms: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # Helper for Pharma Dashboard
    total = Patient.query.count()
    high = Patient.query.filter_by(risk_level='High').count()
    
    # Avg age
    ages = [p.age for p in Patient.query.with_entities(Patient.age).all()]
    avg_age = sum(ages) / len(ages) if ages else 0
    
    return jsonify({
        'totalReports': total,
        'highRiskCount': high,
        'avgAge': round(avg_age),
        'genderDist': {
            'male': Patient.query.filter_by(gender='Male').count(),
            'female': Patient.query.filter_by(gender='Female').count(),
            'other': Patient.query.filter_by(gender='Other').count()
        },
        'riskDist': {
            'low': Patient.query.filter_by(risk_level='Low').count(),
            'medium': Patient.query.filter_by(risk_level='Medium').count(),
            'high': Patient.query.filter_by(risk_level='High').count()
        }
    })

# --- Drug Management API ---

@app.route('/api/drugs', methods=['GET'])
def get_drugs():
    try:
        if session.get('role') != 'pharma':
            return jsonify({'error': 'Unauthorized'}), 403
        
        user_id = session.get('user_id')
        drugs = Drug.query.filter_by(company_id=user_id).all()
        return jsonify([d.to_dict() for d in drugs])
    except Exception as e:
        print(f"ERROR getting drugs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/drugs', methods=['POST'])
def create_drug():
    if session.get('role') != 'pharma':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        data = request.json
        
        # Simulate AI risk assessment
        ai_risk = 'Low'
        ai_details = 'Initial analysis shows standard safety profile.'
        
        # Simple AI logic based on ingredients
        ingredients = data.get('activeIngredients', '').lower()
        if 'pseudo' in ingredients or 'advilone' in ingredients:
            ai_risk = 'High'
            ai_details = 'Historical data shows increased risk with similar compounds. Recommend enhanced monitoring.'
        elif len(ingredients) > 50:
            ai_risk = 'Medium'
            ai_details = 'Complex formulation requires additional safety monitoring.'
        
        new_drug = Drug(
            name=data['name'],
            company_id=session['user_id'],
            description=data.get('description'),
            active_ingredients=data.get('activeIngredients'),
            ai_risk_assessment=ai_risk,
            ai_risk_details=ai_details
        )
        
        db.session.add(new_drug)
        db.session.commit()
        return jsonify({'success': True, 'drug': new_drug.to_dict()})
    except Exception as e:
        print(f"ERROR creating drug: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# --- Alert Management API ---

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        if session.get('role') == 'doctor':
            # Doctors see all alerts
            alerts = Alert.query.order_by(Alert.created_at.desc()).all()
        else:
            # Pharma sees their own sent alerts
            user_id = session.get('user_id')
            alerts = Alert.query.filter_by(sender_id=user_id).order_by(Alert.created_at.desc()).all()
        
        return jsonify([a.to_dict() for a in alerts])
    except Exception as e:
        print(f"ERROR getting alerts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['POST'])
def create_alert():
    if session.get('role') != 'pharma':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        data = request.json
        
        new_alert = Alert(
            drug_name=data['drugName'],
            message=data['message'],
            severity=data.get('severity', 'Medium'),
            sender_id=session['user_id']
        )
        
        db.session.add(new_alert)
        db.session.commit()
        return jsonify({'success': True, 'alert': new_alert.to_dict()})
    except Exception as e:
        print(f"ERROR creating alert: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/alerts/<int:alert_id>/read', methods=['POST'])
def mark_alert_read(alert_id):
    try:
        alert = Alert.query.get(alert_id)
        if not alert:
            return jsonify({'success': False, 'message': 'Alert not found'}), 404
        
        alert.is_read = True
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        print(f"ERROR marking alert read: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
