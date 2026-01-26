"""
Services module for the Pharmacovigilance system.
Contains business logic for case linking, scoring, normalization, follow-ups,
and LLM-based Excel interpretation.
"""

from pv_backend.services.normalization_service import NormalizationService
from pv_backend.services.case_linking_service import CaseLinkingService
from pv_backend.services.scoring_service import ScoringService
from pv_backend.services.followup_service import FollowUpService
from pv_backend.services.audit_service import AuditService

# Optional: Excel LLM service (requires openai package)
try:
    from pv_backend.services.excel_llm_service import ExcelLLMService, ExcelLLMServiceFactory
except ImportError:
    ExcelLLMService = None
    ExcelLLMServiceFactory = None

__all__ = [
    'NormalizationService',
    'CaseLinkingService', 
    'ScoringService',
    'FollowUpService',
    'AuditService',
    'ExcelLLMService',
    'ExcelLLMServiceFactory'
]
