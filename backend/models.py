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

# Association Table
doctor_patient = db.Table('doctor_patient',
    db.Column('doctor_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('patient_id', db.String(20), db.ForeignKey('patient.id'), primary_key=True)
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
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), default='Medium') # Low, Medium, High, Critical
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Pharma company
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    
    sender = db.relationship('User', backref=db.backref('sent_alerts', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'drugName': self.drug_name,
            'message': self.message,
            'severity': self.severity,
            'senderId': self.sender_id,
            'senderName': self.sender.name if self.sender else 'Unknown',
            'created_at': self.created_at.isoformat(),
            'isRead': self.is_read
        }
