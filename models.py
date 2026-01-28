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
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    
    sender = db.relationship('User', backref=db.backref('sent_alerts', lazy=True))
    
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
            'created_at': self.created_at.isoformat(),
            'isRead': self.is_read
        }

class SideEffectReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patient.id'), nullable=False)
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
