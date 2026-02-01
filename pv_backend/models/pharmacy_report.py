"""
Pharmacy Report models - for pharmacy-submitted adverse event reports.
Supports anonymous, identified, and aggregated report types.
"""
import enum
from datetime import datetime
# Use the main app's database
from models import db


class ReportType(enum.Enum):
    """Type of pharmacy report."""
    ANONYMOUS = 'anonymous'
    IDENTIFIED = 'identified'
    AGGREGATED = 'aggregated'


class ReactionSeverity(enum.Enum):
    """Severity of the reported reaction."""
    MILD = 'mild'
    MODERATE = 'moderate'
    SEVERE = 'severe'
    LIFE_THREATENING = 'life_threatening'


class ReactionOutcome(enum.Enum):
    """Outcome of the reported reaction."""
    RECOVERED = 'recovered'
    RECOVERING = 'recovering'
    NOT_RECOVERED = 'not_recovered'
    FATAL = 'fatal'
    UNKNOWN = 'unknown'


class AgeGroup(enum.Enum):
    """Age group categories for aggregated reports."""
    PEDIATRIC = 'pediatric'
    ADOLESCENT = 'adolescent'
    ADULT = 'adult'
    ELDERLY = 'elderly'
    GERIATRIC = 'geriatric'
    UNKNOWN = 'unknown'


class PharmacyReport(db.Model):
    """
    Base pharmacy report model.
    Parent class for different report types.
    """
    __tablename__ = 'pharmacy_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    report_type = db.Column(db.Enum(ReportType), nullable=False)
    
    # Report metadata
    pharmacy_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    report_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Drug information
    drug_name = db.Column(db.String(255), nullable=False)
    drug_batch_number = db.Column(db.String(100))
    drug_expiry_date = db.Column(db.Date)
    
    # Reaction information
    reaction_description = db.Column(db.Text, nullable=False)
    reaction_severity = db.Column(db.Enum(ReactionSeverity))
    reaction_outcome = db.Column(db.Enum(ReactionOutcome))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete
    is_deleted = db.Column(db.Boolean, default=False)
    deleted_at = db.Column(db.DateTime)
    
    # Polymorphic identity
    __mapper_args__ = {
        'polymorphic_identity': 'pharmacy_report',
        'polymorphic_on': report_type
    }
    
    # Relationships
    pharmacy = db.relationship('User', backref='pharmacy_reports')
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'report_type': self.report_type.value if self.report_type else None,
            'drug_name': self.drug_name,
            'reaction_description': self.reaction_description,
            'reaction_severity': self.reaction_severity.value if self.reaction_severity else None,
            'reaction_outcome': self.reaction_outcome.value if self.reaction_outcome else None,
            'report_date': self.report_date.isoformat() if self.report_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AnonymousReport(PharmacyReport):
    """
    Anonymous pharmacy report - no patient identification.
    Used when patient identity cannot be disclosed.
    """
    __tablename__ = 'anonymous_reports'
    
    id = db.Column(db.Integer, db.ForeignKey('pharmacy_reports.id'), primary_key=True)
    
    # Minimal patient info
    age_group = db.Column(db.Enum(AgeGroup))
    gender = db.Column(db.String(20))
    
    # Report details
    concomitant_medications = db.Column(db.Text)
    medical_history = db.Column(db.Text)
    
    __mapper_args__ = {
        'polymorphic_identity': ReportType.ANONYMOUS,
    }
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        data = super().to_dict()
        data.update({
            'age_group': self.age_group.value if self.age_group else None,
            'gender': self.gender,
        })
        return data


class IdentifiedReport(PharmacyReport):
    """
    Identified pharmacy report - with patient identification.
    Contains full patient details for follow-up.
    """
    __tablename__ = 'identified_reports'
    
    id = db.Column(db.Integer, db.ForeignKey('pharmacy_reports.id'), primary_key=True)
    
    # Patient identification
    patient_name = db.Column(db.String(255))
    patient_age = db.Column(db.Integer)
    patient_gender = db.Column(db.String(20))
    patient_phone = db.Column(db.String(20))
    patient_email = db.Column(db.String(255))
    
    # Minimal patient info for anonymous-like identified reports
    age_group = db.Column(db.Enum(AgeGroup))
    gender = db.Column(db.String(20))
    
    # Healthcare provider info
    prescribing_doctor_name = db.Column(db.String(255))
    prescribing_doctor_phone = db.Column(db.String(20))
    hospital_name = db.Column(db.String(255))
    
    # Additional fields for identified reports
    internal_case_id = db.Column(db.String(100))
    treating_hospital_reference = db.Column(db.String(255))
    treating_doctor_name = db.Column(db.String(255))
    
    # Detailed medical info
    indication = db.Column(db.Text)
    concomitant_medications = db.Column(db.Text)
    medical_history = db.Column(db.Text)
    allergies = db.Column(db.Text)
    
    # Follow-up
    follow_up_required = db.Column(db.Boolean, default=False)
    follow_up_date = db.Column(db.Date)
    follow_up_notes = db.Column(db.Text)
    
    __mapper_args__ = {
        'polymorphic_identity': ReportType.IDENTIFIED,
    }
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        data = super().to_dict()
        data.update({
            'patient_name': self.patient_name,
            'patient_age': self.patient_age,
            'patient_gender': self.patient_gender,
            'hospital_name': self.hospital_name,
            'follow_up_required': self.follow_up_required,
        })
        return data


class AggregatedReport(PharmacyReport):
    """
    Aggregated pharmacy report - summary of multiple similar reports.
    Used for trend analysis and pattern detection.
    """
    __tablename__ = 'aggregated_reports'
    
    id = db.Column(db.Integer, db.ForeignKey('pharmacy_reports.id'), primary_key=True)
    
    # Aggregation details
    report_count = db.Column(db.Integer, default=1)
    time_period_start = db.Column(db.Date)
    time_period_end = db.Column(db.Date)
    
    # Demographics
    age_group_distribution = db.Column(db.JSON)  # {'pediatric': 5, 'adult': 10, ...}
    gender_distribution = db.Column(db.JSON)  # {'male': 8, 'female': 7, ...}
    
    # Reaction patterns
    common_reactions = db.Column(db.JSON)  # List of common reactions
    severity_distribution = db.Column(db.JSON)  # {'mild': 5, 'moderate': 8, ...}
    outcome_distribution = db.Column(db.JSON)  # {'recovered': 10, 'recovering': 3, ...}
    
    # Trend analysis
    trend_direction = db.Column(db.String(50))  # increasing, decreasing, stable
    trend_percentage = db.Column(db.Float)
    
    __mapper_args__ = {
        'polymorphic_identity': ReportType.AGGREGATED,
    }
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        data = super().to_dict()
        data.update({
            'report_count': self.report_count,
            'time_period_start': self.time_period_start.isoformat() if self.time_period_start else None,
            'time_period_end': self.time_period_end.isoformat() if self.time_period_end else None,
            'trend_direction': self.trend_direction,
        })
        return data
