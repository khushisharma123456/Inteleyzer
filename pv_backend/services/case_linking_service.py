"""
Case Linking Service for the Pharmacovigilance system.
Deterministic logic to link experience events to existing cases or create new ones.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

from pv_backend.models import (
    db, ExperienceEvent, NormalizedExperience, CaseMaster,
    CaseLinkingLog, AuditLog, AuditAction
)


class CaseLinkingService:
    """
    Service for linking experience events to cases.
    
    PV Context:
    - A case represents a DRUG + PERSON combination
    - Events are linked to cases based on:
      1. Matching patient identifier
      2. Matching drug (normalized)
      3. Time window (configurable)
    - New case is created if no match found
    - Confidence score reflects linking quality
    """
    
    ALGORITHM_VERSION = '1.0.0'
    
    # Time window for case linking (events within this window link to same case)
    DEFAULT_TIME_WINDOW_DAYS = 365
    
    @staticmethod
    def generate_case_number() -> str:
        """
        Generate a unique, human-readable case number.
        
        Format: PV-YYYYMMDD-XXXXX (e.g., PV-20260122-A1B2C)
        """
        date_part = datetime.utcnow().strftime('%Y%m%d')
        unique_part = uuid.uuid4().hex[:5].upper()
        return f'PV-{date_part}-{unique_part}'
    
    @classmethod
    def find_matching_case(cls, 
                          drug_normalized: str,
                          patient_identifier: str,
                          event_date: datetime = None,
                          time_window_days: int = None) -> Tuple[Optional[CaseMaster], float, Dict]:
        """
        Find an existing case that matches the drug-patient combination.
        
        PV Context:
        - Case matching is CRITICAL for accurate pharmacovigilance
        - We use deterministic matching (not probabilistic)
        - Confidence reflects the quality of the match
        
        Args:
            drug_normalized: Normalized drug name
            patient_identifier: Hashed patient identifier
            event_date: Date of the event (for time window check)
            time_window_days: Maximum days between events in same case
        
        Returns:
            Tuple of (case, confidence, criteria_dict)
        """
        if not drug_normalized or not patient_identifier:
            return None, 0.0, {'error': 'Missing drug or patient identifier'}
        
        time_window = time_window_days or cls.DEFAULT_TIME_WINDOW_DAYS
        
        # Query for matching cases
        query = CaseMaster.query.filter(
            CaseMaster.drug_name_normalized == drug_normalized,
            CaseMaster.patient_identifier_hash == patient_identifier,
            CaseMaster.is_deleted == False
        )
        
        # Apply time window if event date is provided
        if event_date:
            cutoff_date = event_date - timedelta(days=time_window)
            query = query.filter(CaseMaster.latest_event_date >= cutoff_date)
        
        # Order by latest event date to get most recent case
        matching_case = query.order_by(CaseMaster.latest_event_date.desc()).first()
        
        if matching_case:
            # Calculate confidence based on match quality
            criteria = {
                'patient_match': True,
                'drug_match': True,
                'time_window_days': time_window,
                'within_time_window': True
            }
            
            # Higher confidence for more complete data
            confidence = 0.9
            if matching_case.event_count > 1:
                confidence = 0.95  # Case has history
            
            return matching_case, confidence, criteria
        
        return None, 0.0, {'patient_match': False, 'drug_match': False}
    
    @classmethod
    def create_case(cls, 
                   drug_normalized: str,
                   drug_active_ingredient: str,
                   patient_identifier: str,
                   event_date: datetime = None) -> CaseMaster:
        """
        Create a new case for a drug-patient combination.
        
        PV Context:
        - New case is created when no matching case exists
        - Case number is unique and human-readable
        - Initial score is 0 (neutral)
        """
        case = CaseMaster(
            case_number=cls.generate_case_number(),
            drug_name_normalized=drug_normalized,
            drug_active_ingredient=drug_active_ingredient,
            patient_identifier_hash=patient_identifier,
            current_score=0.0,
            status='open',
            priority='normal',
            first_event_date=event_date or datetime.utcnow(),
            latest_event_date=event_date or datetime.utcnow(),
            event_count=0
        )
        
        db.session.add(case)
        db.session.flush()  # Flush to get the case.id immediately
        return case
    
    @classmethod
    def link_event_to_case(cls, 
                          event: ExperienceEvent,
                          normalized: NormalizedExperience,
                          user=None) -> Tuple[CaseMaster, CaseLinkingLog]:
        """
        Link an experience event to a case (existing or new).
        
        PV Context:
        - This is the main entry point for case linking
        - Creates a linking log for audit trail
        - Updates case metadata (event count, dates)
        
        Args:
            event: The experience event to link
            normalized: The normalized experience
            user: User performing the action (for audit)
        
        Returns:
            Tuple of (case, linking_log)
        """
        # Find matching case
        matching_case, confidence, criteria = cls.find_matching_case(
            drug_normalized=normalized.drug_name_normalized,
            patient_identifier=normalized.patient_identifier_normalized,
            event_date=event.event_date or event.created_at
        )
        
        is_new_case = False
        
        if matching_case:
            case = matching_case
            # Update case metadata
            case.latest_event_date = event.event_date or event.created_at
            case.event_count += 1
        else:
            # Create new case
            case = cls.create_case(
                drug_normalized=normalized.drug_name_normalized,
                drug_active_ingredient=normalized.drug_active_ingredient,
                patient_identifier=normalized.patient_identifier_normalized,
                event_date=event.event_date or event.created_at
            )
            case.event_count = 1
            is_new_case = True
            confidence = 1.0  # New case = 100% confidence
            criteria = {'new_case': True, 'reason': 'No matching case found'}
        
        # Link event to case
        event.case_id = case.id
        event.is_processed = True
        event.processed_at = datetime.utcnow()
        
        # Create linking log
        linking_log = CaseLinkingLog(
            event_id=event.id,
            case_id=case.id,
            is_new_case=is_new_case,
            confidence=confidence,
            linking_criteria=criteria,
            patient_identifier_matched=criteria.get('patient_match', is_new_case),
            drug_matched=criteria.get('drug_match', is_new_case),
            time_window_days=criteria.get('time_window_days'),
            linked_by='auto',
            linking_algorithm_version=cls.ALGORITHM_VERSION
        )
        
        db.session.add(linking_log)
        
        return case, linking_log
    
    @classmethod
    def override_case_link(cls,
                          event: ExperienceEvent,
                          new_case_id: int,
                          reason: str,
                          user) -> CaseLinkingLog:
        """
        Override an automated case linking decision (human override).
        
        PV Context:
        - Human override is CRITICAL for PV systems
        - Automated decisions can be wrong
        - Full audit trail is maintained
        
        Args:
            event: The event to relink
            new_case_id: ID of the new case to link to
            reason: Reason for the override
            user: User performing the override
        
        Returns:
            Updated linking log
        """
        # Get existing linking log
        existing_log = CaseLinkingLog.query.filter_by(event_id=event.id).first()
        original_case_id = event.case_id
        
        # Get new case
        new_case = CaseMaster.query.get(new_case_id)
        if not new_case:
            raise ValueError(f'Case {new_case_id} not found')
        
        # Update event
        event.case_id = new_case_id
        
        # Update old case event count
        if original_case_id:
            old_case = CaseMaster.query.get(original_case_id)
            if old_case:
                old_case.event_count = max(0, old_case.event_count - 1)
        
        # Update new case event count
        new_case.event_count += 1
        new_case.latest_event_date = max(
            new_case.latest_event_date or datetime.min,
            event.event_date or event.created_at
        )
        
        # Update or create linking log
        if existing_log:
            existing_log.is_overridden = True
            existing_log.override_by_user_id = user.id
            existing_log.override_reason = reason
            existing_log.override_at = datetime.utcnow()
            existing_log.original_case_id = original_case_id
            existing_log.case_id = new_case_id
            linking_log = existing_log
        else:
            linking_log = CaseLinkingLog(
                event_id=event.id,
                case_id=new_case_id,
                is_new_case=False,
                confidence=1.0,
                linking_criteria={'manual_override': True},
                linked_by='manual',
                is_overridden=True,
                override_by_user_id=user.id,
                override_reason=reason,
                override_at=datetime.utcnow(),
                original_case_id=original_case_id
            )
            db.session.add(linking_log)
        
        return linking_log
