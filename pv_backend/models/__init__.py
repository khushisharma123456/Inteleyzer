"""
SQLAlchemy models for the Pharmacovigilance system.
All models are imported here for easy access.
"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

from .user import User, UserRole
from .reporter import Reporter
from .experience_event import ExperienceEvent, EventSource
from .normalized_experience import NormalizedExperience, Polarity
from .case_master import CaseMaster
from .case_score_history import CaseScoreHistory
from .adverse_event import AdverseEvent, Severity, Outcome
from .follow_up_request import FollowUpRequest, FollowUpStatus, FollowUpReason
from .medical_validation import MedicalValidation, ValidationStatus
from .audit_log import AuditLog, AuditAction
from .case_linking_log import CaseLinkingLog
from .pharmacy_report import (
    PharmacyReport, AnonymousReport, IdentifiedReport, AggregatedReport,
    ReportType, ReactionSeverity, ReactionOutcome, AgeGroup
)

__all__ = [
    'db',
    'User', 'UserRole',
    'Reporter',
    'ExperienceEvent', 'EventSource',
    'NormalizedExperience', 'Polarity',
    'CaseMaster',
    'CaseScoreHistory',
    'AdverseEvent', 'Severity', 'Outcome',
    'FollowUpRequest', 'FollowUpStatus', 'FollowUpReason',
    'MedicalValidation', 'ValidationStatus',
    'AuditLog', 'AuditAction',
    'CaseLinkingLog',
    'PharmacyReport', 'AnonymousReport', 'IdentifiedReport', 'AggregatedReport',
    'ReportType', 'ReactionSeverity', 'ReactionOutcome', 'AgeGroup'
]
