"""
SQLAlchemy models for the Pharmacovigilance system.
All models are imported here for easy access.
"""
# Use the main app's database
from models import db

# Import from pharmacy_report module
from .pharmacy_report import (
    PharmacyReport, AnonymousReport, IdentifiedReport, AggregatedReport,
    ReportType, ReactionSeverity, ReactionOutcome, AgeGroup
)

__all__ = [
    'db',
    'PharmacyReport', 'AnonymousReport', 'IdentifiedReport', 'AggregatedReport',
    'ReportType', 'ReactionSeverity', 'ReactionOutcome', 'AgeGroup'
]
