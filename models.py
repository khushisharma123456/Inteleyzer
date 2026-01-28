from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False) # In real app, hash this!
    role = db.Column(db.String(20), nullable=False) # 'doctor' or 'pharma'
    hospital_name = db.Column(db.String(200), nullable=True) # Hospital name for hospital users

# Association Tables
doctor_patient = db.Table('doctor_patient',
    db.Column('doctor_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('patient_id', db.String(20), db.ForeignKey('patient.id'), primary_key=True)
)

hospital_doctor = db.Table('hospital_doctor',
    db.Column('hospital_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('doctor_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

hospital_drug = db.Table('hospital_drug',
    db.Column('hospital_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('drug_id', db.Integer, db.ForeignKey('drug.id'), primary_key=True)
)

hospital_pharmacy = db.Table('hospital_pharmacy',
    db.Column('hospital_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('pharmacy_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

class Patient(db.Model):
    id = db.Column(db.String(20), primary_key=True) # Custom ID like PT-1234
    
    # Many-to-Many with Doctors
    doctors = db.relationship('User', secondary=doctor_patient, backref=db.backref('patients', lazy=True))
    
    # Creator (Optional, for tracking who first made it)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Demographics
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    
    # Clinical
    drug_name = db.Column(db.String(100), nullable=False)
    symptoms = db.Column(db.Text, nullable=True)
    risk_level = db.Column(db.String(20), default='Low') # Low, Medium, High
    
    # Case Linkage & Deduplication
    linked_case_id = db.Column(db.String(20), db.ForeignKey('patient.id'), nullable=True) # Links to parent case if duplicate
    match_score = db.Column(db.Float, nullable=True) # Similarity score with linked case (0-1)
    case_status = db.Column(db.String(20), default='Active') # Active, Linked, Discarded
    match_notes = db.Column(db.Text, nullable=True) # Reason for linkage/discarding
    
    # Patient Recall for Testing
    recalled = db.Column(db.Boolean, default=False) # Whether patient has been recalled for tests
    recalled_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Company that recalled
    recall_reason = db.Column(db.Text, nullable=True) # Reason for recall
    recall_date = db.Column(db.DateTime, nullable=True) # When patient was recalled
    
    # === CASE STRENGTH EVALUATION (STEP 7) ===
    # Completeness Assessment
    completeness_score = db.Column(db.Float, default=0.0)  # 0-1 (0-100%)
    mandatory_fields_filled = db.Column(db.Integer, default=0)
    total_mandatory_fields = db.Column(db.Integer, default=7)
    
    # Temporal Clarity
    temporal_clarity_score = db.Column(db.Float, default=0.0)  # 0-1
    symptom_onset_date = db.Column(db.DateTime, nullable=True)
    symptom_resolution_date = db.Column(db.DateTime, nullable=True)
    has_clear_timeline = db.Column(db.Boolean, default=False)
    
    # Medical Confirmation
    medical_confirmation_score = db.Column(db.Float, default=0.0)  # 0-1
    doctor_confirmed = db.Column(db.Boolean, default=False)
    doctor_confirmation_date = db.Column(db.DateTime, nullable=True)
    hospital_confirmed = db.Column(db.Boolean, default=False)
    hospital_confirmation_date = db.Column(db.DateTime, nullable=True)
    
    # Follow-up Responsiveness
    followup_responsiveness_score = db.Column(db.Float, default=0.0)  # 0-1
    last_followup_date = db.Column(db.DateTime, nullable=True)
    followup_response_time_days = db.Column(db.Integer, nullable=True)
    followup_response_quality = db.Column(db.String(20), nullable=True)  # Poor, Fair, Good
    
    # Overall Strength
    strength_level = db.Column(db.String(20), default='Not Evaluated')  # Low, Medium, High, Not Evaluated
    strength_score = db.Column(db.Integer, default=0)  # 0, 1, 2
    
    # === FINAL CASE SCORING (STEP 8) ===
    polarity = db.Column(db.Integer, nullable=True)  # -1 (AE), 0 (unclear), +1 (positive)
    case_score = db.Column(db.Integer, nullable=True)  # -2 to +2
    case_score_interpretation = db.Column(db.String(100), nullable=True)
    case_score_calculated_at = db.Column(db.DateTime, nullable=True)
    
    # === FOLLOW-UP TRACKING (STEP 9) ===
    follow_up_required = db.Column(db.Boolean, default=False)
    follow_up_sent = db.Column(db.Boolean, default=False)
    follow_up_date = db.Column(db.DateTime, nullable=True)
    follow_up_response = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    evaluated_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'age': self.age,
            'gender': self.gender,
            'drugName': self.drug_name,
            'symptoms': self.symptoms,
            'riskLevel': self.risk_level,
            'recalled': self.recalled,
            'recallReason': self.recall_reason,
            'recallDate': self.recall_date.isoformat() if self.recall_date else None,
            'created_at': self.created_at.isoformat()
        }

class Drug(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    company_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Pharma company
    description = db.Column(db.Text, nullable=True)
    active_ingredients = db.Column(db.Text, nullable=True)
    ai_risk_assessment = db.Column(db.String(20), default='Analyzing') # Analyzing, Low, Medium, High
    ai_risk_details = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    company = db.relationship('User', backref=db.backref('drugs', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'companyId': self.company_id,
            'description': self.description,
            'activeIngredients': self.active_ingredients,
            'aiRiskAssessment': self.ai_risk_assessment,
            'aiRiskDetails': self.ai_risk_details,
            'created_at': self.created_at.isoformat()
        }

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    drug_name = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(200), nullable=True)
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), default='Medium') # Low, Medium, High, Critical
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Pharma company
    recipient_type = db.Column(db.String(20), default='all') # 'all', 'doctors', 'hospitals'
    status = db.Column(db.String(20), default='new') # new, acknowledged, resolved
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    acknowledged_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Pharmacy that acknowledged
    is_read = db.Column(db.Boolean, default=False)
    
    sender = db.relationship('User', foreign_keys=[sender_id], backref=db.backref('sent_alerts', lazy=True))
    acknowledger = db.relationship('User', foreign_keys=[acknowledged_by], backref=db.backref('acknowledged_alerts', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'drug_name': self.drug_name,
            'drugName': self.drug_name,
            'title': self.title,
            'message': self.message,
            'severity': self.severity,
            'sender_id': self.sender_id,
            'senderId': self.sender_id,
            'sender': self.sender.name if self.sender else 'Unknown',
            'senderName': self.sender.name if self.sender else 'Unknown',
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'isRead': self.is_read,
            'type': 'safety',  # Default type
            'reason': 'Safety monitoring',  # Default reason
            'impact': 'Review recommended'  # Default impact
        }

class SideEffectReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patient.id'), nullable=True)  # Nullable for anonymised reports
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    hospital_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Hospital if doctor is registered
    drug_name = db.Column(db.String(100), nullable=False)
    drug_id = db.Column(db.Integer, db.ForeignKey('drug.id'), nullable=True)
    side_effect = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), default='Medium') # Low, Medium, High, Critical
    company_notified = db.Column(db.Boolean, default=False)
    hospital_notified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    patient = db.relationship('Patient', backref=db.backref('side_effect_reports', lazy=True))
    doctor = db.relationship('User', foreign_keys=[doctor_id], backref=db.backref('reported_side_effects', lazy=True))
    hospital = db.relationship('User', foreign_keys=[hospital_id], backref=db.backref('received_side_effect_reports', lazy=True))
    drug = db.relationship('Drug', backref=db.backref('side_effect_reports', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'doctor_id': self.doctor_id,
            'hospital_id': self.hospital_id,
            'drug_name': self.drug_name,
            'side_effect': self.side_effect,
            'severity': self.severity,
            'company_notified': self.company_notified,
            'hospital_notified': self.hospital_notified,
            'created_at': self.created_at.isoformat(),
            'doctor_name': self.doctor.name if self.doctor else 'Unknown',
            'hospital_name': self.hospital.name if self.hospital else 'N/A'
        }


class PharmacySettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pharmacy_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    
    # Account settings
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.String(200), nullable=True)
    license = db.Column(db.String(100), nullable=True)
    
    # Privacy settings
    share_reports = db.Column(db.Boolean, default=True)
    share_dispensing = db.Column(db.Boolean, default=True)
    anonymize_data = db.Column(db.Boolean, default=False)
    retention_period = db.Column(db.String(10), default='12')
    
    # Notification settings
    alert_frequency = db.Column(db.String(20), default='immediate')
    notify_email = db.Column(db.Boolean, default=True)
    notify_sms = db.Column(db.Boolean, default=False)
    notify_dashboard = db.Column(db.Boolean, default=True)
    alert_recalls = db.Column(db.Boolean, default=True)
    alert_safety = db.Column(db.Boolean, default=True)
    alert_interactions = db.Column(db.Boolean, default=True)
    alert_dosage = db.Column(db.Boolean, default=True)
    
    # Compliance settings
    reporting_authority = db.Column(db.String(100), nullable=True)
    reporting_threshold = db.Column(db.String(20), default='all')
    compliance_officer = db.Column(db.String(120), nullable=True)
    auto_report = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    pharmacy = db.relationship('User', backref=db.backref('settings', uselist=False))
    
    def to_dict(self):
        return {
            'pharmacyName': self.pharmacy.name if self.pharmacy else '',
            'email': self.pharmacy.email if self.pharmacy else '',
            'phone': self.phone or '',
            'address': self.address or '',
            'license': self.license or '',
            'shareReports': self.share_reports,
            'shareDispensing': self.share_dispensing,
            'anonymizeData': self.anonymize_data,
            'retentionPeriod': self.retention_period,
            'alertFrequency': self.alert_frequency,
            'notifyEmail': self.notify_email,
            'notifySms': self.notify_sms,
            'notifyDashboard': self.notify_dashboard,
            'alertRecalls': self.alert_recalls,
            'alertSafety': self.alert_safety,
            'alertInteractions': self.alert_interactions,
            'alertDosage': self.alert_dosage,
            'reportingAuthority': self.reporting_authority or '',
            'reportingThreshold': self.reporting_threshold,
            'complianceOfficer': self.compliance_officer or '',
            'autoReport': self.auto_report
        }


# === STEP 10: AI AGENT ORCHESTRATION ===

class CaseAgent(db.Model):
    """
    AI Agents that improve case quality by requesting additional information
    Agents: Patient (symptom clarity), Doctor (medical confirmation), Hospital (clinical records)
    """
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.String(20), db.ForeignKey('patient.id'), nullable=False)
    agent_type = db.Column(db.String(20), nullable=False)  # 'patient', 'doctor', 'hospital'
    role = db.Column(db.String(100))  # Role description
    target_questions = db.Column(db.JSON)  # List of questions to ask
    recipient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    status = db.Column(db.String(20), default='active')  # active, completed, failed
    responses = db.Column(db.JSON, nullable=True)  # Responses received
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime, nullable=True)
    
    case = db.relationship('Patient', backref=db.backref('quality_agents', lazy=True))
    recipient = db.relationship('User', backref=db.backref('assigned_agents', lazy=True))


class FollowUp(db.Model):
    """
    Track all follow-up activities to improve case quality
    """
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.String(20), db.ForeignKey('patient.id'), nullable=False)
    reason = db.Column(db.Text, nullable=False)  # Why follow-up is needed
    assigned_to = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, resolved
    priority = db.Column(db.String(20), default='medium')  # low, medium, high, critical
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime, nullable=True)
    response = db.Column(db.Text, nullable=True)
    
    case = db.relationship('Patient', backref=db.backref('followups', lazy=True))
    assigned_user = db.relationship('User', backref=db.backref('assigned_followups', lazy=True))
