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

@app.route('/doctor/report')
def doctor_report():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login_page'))
    return render_template('doctor/report.html')

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

@app.route('/pharma/analysis')
def pharma_analysis():
    if 'user_id' not in session or session.get('role') != 'pharma':
        return redirect(url_for('login_page'))
    return render_template('pharma/analysis.html')

@app.route('/doctor/analysis')
def doctor_analysis():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login_page'))
    return render_template('doctor/analysis.html')

# --- Pharmacy Routes ---

@app.route('/pharmacy/dashboard')
def pharmacy_dashboard():
    if 'user_id' not in session or session.get('role') != 'pharmacy':
        return redirect(url_for('login_page'))
    return render_template('pharmacy/dashboard.html')

@app.route('/pharmacy/reports')
def pharmacy_reports():
    if 'user_id' not in session or session.get('role') != 'pharmacy':
        return redirect(url_for('login_page'))
    return render_template('pharmacy/reports.html')

@app.route('/pharmacy/report')
def pharmacy_report_form():
    if 'user_id' not in session or session.get('role') != 'pharmacy':
        return redirect(url_for('login_page'))
    return render_template('pharmacy/report.html')

@app.route('/pharmacy/alerts')
def pharmacy_alerts():
    if 'user_id' not in session or session.get('role') != 'pharmacy':
        return redirect(url_for('login_page'))
    return render_template('pharmacy/alerts.html')

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
        
        # Both pharma and doctors can now see all patients
        patients = Patient.query.all()
            
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

@app.route('/api/analytics/advanced', methods=['GET'])
def get_advanced_analytics():
    """Comprehensive analytics for the Analysis page"""
    try:
        patients = Patient.query.all()
        drugs = Drug.query.all()
        alerts = Alert.query.all()
        
        # Age distribution by ranges
        age_ranges = {'18-30': 0, '31-40': 0, '41-50': 0, '51-60': 0, '61-70': 0, '71+': 0}
        for p in patients:
            if p.age <= 30:
                age_ranges['18-30'] += 1
            elif p.age <= 40:
                age_ranges['31-40'] += 1
            elif p.age <= 50:
                age_ranges['41-50'] += 1
            elif p.age <= 60:
                age_ranges['51-60'] += 1
            elif p.age <= 70:
                age_ranges['61-70'] += 1
            else:
                age_ranges['71+'] += 1
        
        # Top drugs by adverse events
        drug_counts = {}
        for p in patients:
            drug_counts[p.drug_name] = drug_counts.get(p.drug_name, 0) + 1
        top_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Risk by drug
        risk_by_drug = {}
        for p in patients:
            if p.drug_name not in risk_by_drug:
                risk_by_drug[p.drug_name] = {'Low': 0, 'Medium': 0, 'High': 0}
            risk_by_drug[p.drug_name][p.risk_level] += 1
        
        # Gender vs Risk analysis
        gender_risk = {
            'Male': {'Low': 0, 'Medium': 0, 'High': 0},
            'Female': {'Low': 0, 'Medium': 0, 'High': 0},
            'Other': {'Low': 0, 'Medium': 0, 'High': 0}
        }
        for p in patients:
            gender_risk[p.gender][p.risk_level] += 1
        
        # Age vs Risk analysis
        age_risk = {'Low': [], 'Medium': [], 'High': []}
        for p in patients:
            age_risk[p.risk_level].append(p.age)
        
        age_risk_avg = {
            'Low': round(sum(age_risk['Low']) / len(age_risk['Low'])) if age_risk['Low'] else 0,
            'Medium': round(sum(age_risk['Medium']) / len(age_risk['Medium'])) if age_risk['Medium'] else 0,
            'High': round(sum(age_risk['High']) / len(age_risk['High'])) if age_risk['High'] else 0
        }
        
        # Alert severity distribution
        alert_severity = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
        for a in alerts:
            alert_severity[a.severity] = alert_severity.get(a.severity, 0) + 1
        
        # Drug risk assessment distribution
        drug_risk_dist = {'Low': 0, 'Medium': 0, 'High': 0}
        for d in drugs:
            drug_risk_dist[d.ai_risk_assessment] = drug_risk_dist.get(d.ai_risk_assessment, 0) + 1
        
        # Monthly trend (last 6 months)
        from datetime import datetime, timedelta
        monthly_reports = {}
        for p in patients:
            month_key = p.created_at.strftime('%Y-%m')
            monthly_reports[month_key] = monthly_reports.get(month_key, 0) + 1
        
        # Sort by month
        monthly_trend = sorted(monthly_reports.items())
        
        return jsonify({
            'success': True,
            'ageDistribution': age_ranges,
            'topDrugsByReports': [{'drug': d[0], 'count': d[1]} for d in top_drugs],
            'riskByDrug': risk_by_drug,
            'genderRiskAnalysis': gender_risk,
            'ageRiskAverage': age_risk_avg,
            'alertSeverityDistribution': alert_severity,
            'drugRiskDistribution': drug_risk_dist,
            'monthlyTrend': [{'month': m[0], 'count': m[1]} for m in monthly_trend],
            'totalPatients': len(patients),
            'totalDrugs': len(drugs),
            'totalAlerts': len(alerts)
        })
    except Exception as e:
        print(f"ERROR in advanced analytics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

# --- Excel Upload and Analysis API ---

@app.route('/api/excel/analyze', methods=['POST'])
def analyze_excel():
    """Analyze uploaded Excel file for missing PV data fields"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'success': False, 'message': 'Invalid file type. Use .xlsx or .xls'}), 400
    
    try:
        import pandas as pd
        
        # Read Excel file
        df = pd.read_excel(file)
        
        # Define required and optional fields for PV
        required_fields = ['patient_identifier', 'drug_name']
        optional_fields = ['drug_code', 'indication', 'dosage', 'route', 'start_date', 
                          'event_date', 'observed_events', 'outcome']
        
        # Normalize column names (handle both underscore and space versions)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        total_records = len(df)
        analysis = {
            'total_records': total_records,
            'complete_records': 0,
            'records_with_missing_required': 0,
            'records_with_missing_optional': 0,
            'missing_by_field': {},
            'field_completeness': {},
            'preview_data': df.head(50).to_dict(orient='records')
        }
        
        # Analyze each field
        all_fields = required_fields + optional_fields
        for field in all_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum() + (df[field] == '').sum()
                analysis['missing_by_field'][field] = int(missing_count)
                analysis['field_completeness'][field] = round((total_records - missing_count) / total_records * 100, 1)
            else:
                analysis['missing_by_field'][field] = total_records
                analysis['field_completeness'][field] = 0.0
        
        # Count records by completeness
        for idx, row in df.iterrows():
            has_all_required = True
            missing_optional = 0
            
            for field in required_fields:
                if field not in df.columns or pd.isna(row.get(field)) or row.get(field) == '':
                    has_all_required = False
                    break
            
            for field in optional_fields:
                if field not in df.columns or pd.isna(row.get(field)) or row.get(field) == '':
                    missing_optional += 1
            
            if has_all_required:
                analysis['complete_records'] += 1
            else:
                analysis['records_with_missing_required'] += 1
            
            if missing_optional > 0:
                analysis['records_with_missing_optional'] += 1
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        print(f"ERROR analyzing Excel: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/excel/template', methods=['GET'])
def get_excel_template():
    """Generate and download PV data template"""
    try:
        import pandas as pd
        from io import BytesIO
        from flask import send_file
        
        # Create template with sample data
        template_data = {
            'patient_identifier': ['P001', 'P002'],
            'drug_name': ['Aspirin', 'Ibuprofen'],
            'drug_code': ['NDC-12345', 'NDC-67890'],
            'indication': ['Pain management', 'Inflammation'],
            'dosage': ['500mg daily', '200mg twice daily'],
            'route': ['oral', 'oral'],
            'start_date': ['2024-01-01', '2024-01-15'],
            'event_date': ['2024-01-15', '2024-01-20'],
            'observed_events': ['No adverse events', 'Mild nausea'],
            'outcome': ['ongoing', 'recovered']
        }
        
        df = pd.DataFrame(template_data)
        
        # Save to bytes buffer
        output = BytesIO()
        df.to_excel(output, index=False, sheet_name='Drug Experiences')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='pv_template.xlsx'
        )
        
    except Exception as e:
        print(f"ERROR generating template: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# --- Pharmacy API Endpoints ---

@app.route('/api/pharmacy/stats', methods=['GET'])
def get_pharmacy_stats():
    """Get pharmacy dashboard statistics"""
    try:
        pharmacy_id = session.get('user_id')
        
        # Get all reports from this pharmacy (stored in created_by field)
        all_reports = Patient.query.filter_by(created_by=pharmacy_id).all()
        
        # Today's reports
        from datetime import date
        today_reports = [r for r in all_reports if r.created_at.date() == date.today()]
        
        # Get alerts
        high_alerts = Alert.query.filter(Alert.severity.in_(['High', 'Critical'])).filter_by(is_read=False).count()
        
        # Mock dispensing count (in production, this would come from a separate table)
        dispensing_count = random.randint(150, 300)
        
        # Severity distribution
        severity_counts = {
            'low': len([r for r in all_reports if r.risk_level == 'Low']),
            'medium': len([r for r in all_reports if r.risk_level == 'Medium']),
            'high': len([r for r in all_reports if r.risk_level == 'High'])
        }
        
        # Top drugs by reports
        drug_counts = {}
        for report in all_reports:
            drug_counts[report.drug_name] = drug_counts.get(report.drug_name, 0) + 1
        
        top_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_drugs_list = [{'name': d[0], 'count': d[1]} for d in top_drugs]
        
        return jsonify({
            'success': True,
            'today': len(today_reports),
            'total': len(all_reports),
            'alerts': high_alerts,
            'dispensing': dispensing_count,
            'severity': severity_counts,
            'topDrugs': top_drugs_list
        })
    except Exception as e:
        print(f"ERROR getting pharmacy stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pharmacy/reports', methods=['GET'])
def get_pharmacy_reports():
    """Get all ADR reports from this pharmacy"""
    try:
        pharmacy_id = session.get('user_id')
        limit = request.args.get('limit', type=int)
        
        # Get reports created by this pharmacy
        query = Patient.query.filter_by(created_by=pharmacy_id).order_by(Patient.created_at.desc())
        
        if limit:
            query = query.limit(limit)
        
        reports = query.all()
        
        reports_data = [{
            'id': r.id,
            'date': r.created_at.isoformat(),
            'patientName': r.name,
            'age': r.age,
            'gender': r.gender,
            'drugName': r.drug_name,
            'reaction': r.symptoms,
            'severity': r.risk_level,
            'status': 'Submitted',  # Could be enhanced with a status field
            'notes': ''
        } for r in reports]
        
        return jsonify({
            'success': True,
            'reports': reports_data
        })
    except Exception as e:
        print(f"ERROR getting pharmacy reports: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pharmacy/report', methods=['POST'])
def submit_pharmacy_report():
    """Submit a new ADR report from pharmacy"""
    try:
        pharmacy_id = session.get('user_id')
        data = request.json
        
        # Determine risk level based on severity
        risk_level = data.get('severity', 'Low')
        
        # Create new patient/report entry
        patient = Patient(
            id='PH-' + str(random.randint(1000, 9999)),
            created_by=pharmacy_id,
            name=data['patientName'],
            phone=data.get('phone', ''),
            age=data['age'],
            gender=data['gender'],
            drug_name=data['drugName'],
            symptoms=data['reaction'],
            risk_level=risk_level
        )
        
        db.session.add(patient)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'ADR report submitted successfully',
            'report_id': patient.id
        })
    except Exception as e:
        print(f"ERROR submitting pharmacy report: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
