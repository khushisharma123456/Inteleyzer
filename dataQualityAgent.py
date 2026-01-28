"""
📋 DATA QUALITY & CONFIDENCE CONTROL AGENT (Agent 2)

This module implements Agent 2: a strict data quality gatekeeper for the
pharmacovigilance WhatsApp follow-up system.

CORE PRINCIPLES (NON-NEGOTIABLE):
1. Missing data is BETTER than wrong data
2. NEVER guess, autofill, or modify patient data
3. Only reads, validates, and flags issues
4. Works alongside Agent 1 (Conversational Agent)

DATA QUALITY STATUSES:
- PASS: All data valid and complete
- NEEDS_COMPLETION: Missing required fields, clarification needed
- LOW_CONFIDENCE: Conflicting data or abandoned conversation

Author: Auto-generated based on implementation plan
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS & TYPE DEFINITIONS
# =============================================================================

class DataQualityStatus(Enum):
    """Possible data quality statuses for a visit."""
    PASS = "PASS"
    NEEDS_COMPLETION = "NEEDS_COMPLETION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class VisitType(Enum):
    """Types of patient visits."""
    NEW = "new"
    FOLLOW_UP = "follow-up"
    SIDE_EFFECT = "side-effect"


class ClarificationStatus(Enum):
    """Status of a clarification request."""
    PENDING = "pending"
    SENT = "sent"
    ANSWERED = "answered"
    EXPIRED = "expired"


class DoctorEntry(TypedDict):
    """Type definition for doctor-entered data."""
    visit_id: int
    patient_id: str
    disease_name: str
    medicine_prescribed: str
    visit_date: str
    visit_type: str  # new / follow-up / side-effect
    dosage: Optional[str]
    frequency: Optional[str]
    duration: Optional[str]
    diagnosis_notes: Optional[str]


class PatientResponse(TypedDict):
    """Type definition for patient response data."""
    response_id: int
    visit_id: int
    question_id: str
    answer: str
    timestamp: str


class Inconsistency(TypedDict):
    """Type definition for detected inconsistencies."""
    field1: str
    field2: str
    conflict: str


class DataQualityReport(TypedDict):
    """Type definition for the quality report output."""
    visit_id: int
    data_quality_status: str
    missing_fields: List[str]
    inconsistent_fields: List[Inconsistency]
    confidence_score: float
    clarification_required: bool
    clarification_question_ids: List[str]


# =============================================================================
# MANDATORY FIELD DEFINITIONS
# =============================================================================

# Doctor entry mandatory fields
MANDATORY_DOCTOR_FIELDS = [
    "patient_id",
    "disease_name",
    "medicine_prescribed",
    "visit_date",
    "visit_type"
]

# Patient response expected questions by visit type
PATIENT_QUESTIONS_NEW_VISIT = {
    "Q1_language": {"required": True, "description": "Language selection"},
    "Q2_medicine_started": {"required": True, "description": "Medicine started"},
    "Q3_adherence": {"required": False, "conditional_on": "Q2_medicine_started", "condition_value": "yes", "description": "Adherence frequency"},
    "Q4_food_relation": {"required": False, "conditional_on": "Q2_medicine_started", "condition_value": "yes", "description": "Food relation"},
    "Q5_new_symptoms": {"required": True, "description": "New symptoms appeared"},
    "Q6_symptom_description": {"required": False, "conditional_on": "Q5_new_symptoms", "condition_value": "yes", "description": "Symptom description"},
    "Q7_severity": {"required": False, "conditional_on": "Q5_new_symptoms", "condition_value": "yes", "description": "Symptom severity"},
    "Q8_body_parts": {"required": False, "conditional_on": "Q5_new_symptoms", "condition_value": "yes", "description": "Affected body parts"},
    "Q9_safety_check": {"required": True, "description": "Safety check confirmation"}
}

PATIENT_QUESTIONS_REVISIT = {
    "Q1_language": {"required": True, "description": "Language selection"},
    "Q2_medicine_continued": {"required": True, "description": "Medicine continued"},
    "Q3_adherence": {"required": False, "conditional_on": "Q2_medicine_continued", "condition_value": "yes", "description": "Adherence frequency"},
    "Q4_food_relation": {"required": False, "conditional_on": "Q2_medicine_continued", "condition_value": "yes", "description": "Food relation"},
    "Q4b_previous_symptoms": {"required": True, "description": "Previous symptoms resolved"},
    "Q5_new_symptoms": {"required": True, "description": "New symptoms appeared"},
    "Q6_symptom_description": {"required": False, "conditional_on": "Q5_new_symptoms", "condition_value": "yes", "description": "Symptom description"},
    "Q7_severity": {"required": False, "conditional_on": "Q5_new_symptoms", "condition_value": "yes", "description": "Symptom severity"},
    "Q8_body_parts": {"required": False, "conditional_on": "Q5_new_symptoms", "condition_value": "yes", "description": "Affected body parts"},
    "Q9_safety_check": {"required": True, "description": "Safety check confirmation"}
}

# Inconsistency detection rules
INCONSISTENCY_RULES = [
    {
        "name": "medicine_adherence_conflict",
        "field1": "Q2_medicine_started",
        "value1": "no",
        "field2": "Q3_adherence",
        "value2_exists": True,
        "conflict_message": "Patient says 'no' to medicine but answered adherence question"
    },
    {
        "name": "symptom_severity_conflict",
        "field1": "Q5_new_symptoms",
        "value1": "no",
        "field2": "Q6_symptom_description",
        "value2_exists": True,
        "conflict_message": "Patient says 'no' to new symptoms but provided symptom description"
    },
    {
        "name": "no_symptom_but_body_parts",
        "field1": "Q5_new_symptoms",
        "value1": "no",
        "field2": "Q8_body_parts",
        "value2_exists": True,
        "conflict_message": "Patient says 'no' to new symptoms but provided affected body parts"
    }
]


# =============================================================================
# DATA QUALITY AGENT CLASS
# =============================================================================

class DataQualityAgent:
    """
    Data Quality & Confidence Control Agent (Agent 2)
    
    Responsibilities:
    1. Validate doctor-entered data completeness
    2. Validate patient responses from Agent 1
    3. Detect inconsistencies across the visit
    4. Assign confidence scores (0-1)
    5. Generate clarification requests for Agent 1
    
    CORE PRINCIPLE: Missing data is better than wrong data.
    Agent 2 must NEVER guess, autofill, or modify patient data.
    """
    
    def __init__(self, visit_id: int):
        """
        Initialize the Data Quality Agent for a specific visit.
        
        Args:
            visit_id: The unique identifier for the visit to analyze
        """
        self.visit_id = visit_id
        self._doctor_data: Optional[DoctorEntry] = None
        self._patient_responses: List[PatientResponse] = []
        self._is_revisit: bool = False
        self._conversation_complete: bool = False
        self._conversation_abandoned: bool = False
        
        # Validation results (computed on demand)
        self._missing_doctor_fields: List[str] = []
        self._missing_patient_fields: List[str] = []
        self._inconsistencies: List[Inconsistency] = []
        self._confidence_score: float = 0.0
        self._validated: bool = False
    
    # -------------------------------------------------------------------------
    # DATA LOADING METHODS
    # -------------------------------------------------------------------------
    
    def load_doctor_data(self, doctor_data: DoctorEntry) -> None:
        """
        Load doctor-entered data for validation.
        
        Args:
            doctor_data: Dictionary containing doctor entry fields
        """
        self._doctor_data = doctor_data
        self._is_revisit = doctor_data.get("visit_type", "").lower() in ["follow-up", "revisit"]
        self._validated = False  # Reset validation flag
    
    def load_patient_responses(self, responses: List[PatientResponse]) -> None:
        """
        Load patient responses from Agent 1 conversation.
        
        Args:
            responses: List of patient response dictionaries
        """
        self._patient_responses = responses
        self._validated = False  # Reset validation flag
    
    def set_conversation_status(self, complete: bool, abandoned: bool = False) -> None:
        """
        Set the conversation completion status.
        
        Args:
            complete: Whether the conversation completed successfully
            abandoned: Whether the conversation was abandoned
        """
        self._conversation_complete = complete
        self._conversation_abandoned = abandoned
        self._validated = False  # Reset validation flag
    
    # -------------------------------------------------------------------------
    # VALIDATION METHODS
    # -------------------------------------------------------------------------
    
    def validate_doctor_data(self) -> List[str]:
        """
        Check mandatory fields in doctor entries.
        
        Returns:
            List of missing mandatory field names
        """
        missing_fields = []
        
        if not self._doctor_data:
            return MANDATORY_DOCTOR_FIELDS.copy()
        
        for field in MANDATORY_DOCTOR_FIELDS:
            value = self._doctor_data.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                missing_fields.append(field)
        
        self._missing_doctor_fields = missing_fields
        return missing_fields
    
    def validate_patient_responses(self) -> List[str]:
        """
        Detect missing/incomplete patient answers based on visit type.
        
        Returns:
            List of missing question IDs
        """
        missing_questions = []
        
        # Select appropriate question set based on visit type
        expected_questions = PATIENT_QUESTIONS_REVISIT if self._is_revisit else PATIENT_QUESTIONS_NEW_VISIT
        
        # Build a map of answered questions
        answered_map: Dict[str, str] = {}
        for response in self._patient_responses:
            question_id = response.get("question_id", "")
            answer = response.get("answer", "")
            if question_id:
                answered_map[question_id] = answer.lower().strip()
        
        # Check each expected question
        for question_id, config in expected_questions.items():
            is_required = config.get("required", False)
            conditional_on = config.get("conditional_on")
            condition_value = config.get("condition_value", "").lower()
            
            # Determine if this question should be answered
            should_be_answered = is_required
            
            if conditional_on:
                # This is a conditional question
                parent_answer = answered_map.get(conditional_on, "")
                if parent_answer == condition_value:
                    should_be_answered = True
                else:
                    should_be_answered = False
            
            # Check if question was answered
            if should_be_answered:
                answer = answered_map.get(question_id, "")
                if not answer:
                    missing_questions.append(question_id)
        
        self._missing_patient_fields = missing_questions
        return missing_questions
    
    def detect_inconsistencies(self) -> List[Inconsistency]:
        """
        Cross-check for conflicting responses.
        
        Returns:
            List of detected inconsistencies
        """
        inconsistencies = []
        
        # Build answer map
        answered_map: Dict[str, str] = {}
        for response in self._patient_responses:
            question_id = response.get("question_id", "")
            answer = response.get("answer", "")
            if question_id:
                answered_map[question_id] = answer.lower().strip()
        
        # Check each inconsistency rule
        for rule in INCONSISTENCY_RULES:
            field1 = rule["field1"]
            expected_value1 = rule["value1"]
            field2 = rule["field2"]
            check_exists = rule.get("value2_exists", False)
            conflict_message = rule["conflict_message"]
            
            actual_value1 = answered_map.get(field1, "")
            actual_value2 = answered_map.get(field2, "")
            
            # Check if rule is violated
            if actual_value1 == expected_value1:
                if check_exists and actual_value2:
                    inconsistencies.append({
                        "field1": field1,
                        "field2": field2,
                        "conflict": conflict_message
                    })
        
        self._inconsistencies = inconsistencies
        return inconsistencies
    
    # -------------------------------------------------------------------------
    # CONFIDENCE SCORING
    # -------------------------------------------------------------------------
    
    def calculate_confidence_score(self) -> float:
        """
        Compute confidence score (0-1) based on data completeness and consistency.
        
        Scoring Logic:
        - All mandatory doctor fields present: +0.25
        - All expected patient responses received: +0.25
        - No detected inconsistencies: +0.25
        - Conversation completed (not abandoned): +0.25
        - Missing mandatory field: -0.15 per field
        - Detected inconsistency: -0.20 per conflict
        - Abandoned conversation: -0.30
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        
        # Ensure validation has been run
        if not self._validated:
            self._run_full_validation()
        
        # Positive contributions
        if not self._missing_doctor_fields:
            score += 0.25
        
        if not self._missing_patient_fields:
            score += 0.25
        
        if not self._inconsistencies:
            score += 0.25
        
        if self._conversation_complete and not self._conversation_abandoned:
            score += 0.25
        
        # Negative penalties
        score -= len(self._missing_doctor_fields) * 0.15
        score -= len(self._missing_patient_fields) * 0.10  # Slightly lower penalty for patient fields
        score -= len(self._inconsistencies) * 0.20
        
        if self._conversation_abandoned:
            score -= 0.30
        
        # Clamp score between 0 and 1
        self._confidence_score = max(0.0, min(1.0, score))
        return self._confidence_score
    
    # -------------------------------------------------------------------------
    # QUALITY REPORT GENERATION
    # -------------------------------------------------------------------------
    
    def generate_quality_report(self) -> DataQualityReport:
        """
        Produce final output with status & flags.
        
        Returns:
            Complete data quality report dictionary
        """
        # Run full validation if needed
        if not self._validated:
            self._run_full_validation()
        
        # Calculate confidence score
        confidence = self.calculate_confidence_score()
        
        # Determine status
        status = self._determine_status(confidence)
        
        # Determine if clarification is needed
        all_missing = self._missing_doctor_fields + self._missing_patient_fields
        clarification_required = len(all_missing) > 0 or len(self._inconsistencies) > 0
        
        # Get clarification question IDs (only patient-related ones can be clarified via Agent 1)
        clarification_ids = self.get_clarification_requests()
        
        return {
            "visit_id": self.visit_id,
            "data_quality_status": status.value,
            "missing_fields": all_missing,
            "inconsistent_fields": self._inconsistencies,
            "confidence_score": round(confidence, 2),
            "clarification_required": clarification_required,
            "clarification_question_ids": clarification_ids
        }
    
    def get_clarification_requests(self) -> List[str]:
        """
        Return question IDs that need clarification from patient via Agent 1.
        
        Note: Doctor fields cannot be clarified via the patient. Only patient
        response fields are returned here.
        
        Returns:
            List of question IDs to ask the patient
        """
        clarification_ids = []
        
        # Add missing patient response fields
        for question_id in self._missing_patient_fields:
            clarification_ids.append(question_id)
        
        # For inconsistencies, we might need to re-ask related questions
        for inconsistency in self._inconsistencies:
            # Re-ask the second field to clarify the conflict
            field2 = inconsistency.get("field2", "")
            if field2 and field2 not in clarification_ids:
                clarification_ids.append(field2)
        
        return clarification_ids
    
    # -------------------------------------------------------------------------
    # PRIVATE HELPER METHODS
    # -------------------------------------------------------------------------
    
    def _run_full_validation(self) -> None:
        """Run all validation checks."""
        self.validate_doctor_data()
        self.validate_patient_responses()
        self.detect_inconsistencies()
        self._validated = True
    
    def _determine_status(self, confidence: float) -> DataQualityStatus:
        """
        Determine the data quality status based on validation results and confidence.
        
        Args:
            confidence: The calculated confidence score
            
        Returns:
            DataQualityStatus enum value
        """
        # If abandoned or very low confidence, it's LOW_CONFIDENCE
        if self._conversation_abandoned or confidence < 0.4:
            return DataQualityStatus.LOW_CONFIDENCE
        
        # If there are missing fields or inconsistencies, needs completion
        if self._missing_doctor_fields or self._missing_patient_fields or self._inconsistencies:
            return DataQualityStatus.NEEDS_COMPLETION
        
        # Everything looks good
        return DataQualityStatus.PASS


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_visit_quality(
    visit_id: int,
    doctor_data: DoctorEntry,
    patient_responses: List[PatientResponse],
    conversation_complete: bool = True,
    conversation_abandoned: bool = False
) -> DataQualityReport:
    """
    Convenience function to analyze a visit's data quality in one call.
    
    Args:
        visit_id: The unique visit identifier
        doctor_data: Doctor-entered data dictionary
        patient_responses: List of patient response dictionaries
        conversation_complete: Whether the conversation completed
        conversation_abandoned: Whether the conversation was abandoned
        
    Returns:
        Complete data quality report
    """
    agent = DataQualityAgent(visit_id)
    agent.load_doctor_data(doctor_data)
    agent.load_patient_responses(patient_responses)
    agent.set_conversation_status(conversation_complete, conversation_abandoned)
    
    return agent.generate_quality_report()


def get_expected_questions_for_visit_type(is_revisit: bool) -> Dict[str, Dict]:
    """
    Get the expected questions configuration for a visit type.
    
    Args:
        is_revisit: Whether this is a follow-up/revisit
        
    Returns:
        Dictionary of expected questions with their configurations
    """
    return PATIENT_QUESTIONS_REVISIT if is_revisit else PATIENT_QUESTIONS_NEW_VISIT


# =============================================================================
# EXAMPLE USAGE / TESTING
# =============================================================================

if __name__ == "__main__":
    # Example: Complete data scenario
    print("=" * 60)
    print("Test 1: Complete data - Should PASS")
    print("=" * 60)
    
    doctor_data_complete: DoctorEntry = {
        "visit_id": 123,
        "patient_id": "P001",
        "disease_name": "Hypertension",
        "medicine_prescribed": "Amlodipine 5mg",
        "visit_date": "2026-01-08",
        "visit_type": "new",
        "dosage": "5mg",
        "frequency": "Once daily",
        "duration": "30 days",
        "diagnosis_notes": "Blood pressure elevated, starting medication"
    }
    
    patient_responses_complete: List[PatientResponse] = [
        {"response_id": 1, "visit_id": 123, "question_id": "Q1_language", "answer": "english", "timestamp": "2026-01-08T10:00:00"},
        {"response_id": 2, "visit_id": 123, "question_id": "Q2_medicine_started", "answer": "yes", "timestamp": "2026-01-08T10:01:00"},
        {"response_id": 3, "visit_id": 123, "question_id": "Q3_adherence", "answer": "daily", "timestamp": "2026-01-08T10:02:00"},
        {"response_id": 4, "visit_id": 123, "question_id": "Q4_food_relation", "answer": "after food", "timestamp": "2026-01-08T10:03:00"},
        {"response_id": 5, "visit_id": 123, "question_id": "Q5_new_symptoms", "answer": "no", "timestamp": "2026-01-08T10:04:00"},
        {"response_id": 6, "visit_id": 123, "question_id": "Q9_safety_check", "answer": "confirmed", "timestamp": "2026-01-08T10:05:00"}
    ]
    
    report = analyze_visit_quality(
        visit_id=123,
        doctor_data=doctor_data_complete,
        patient_responses=patient_responses_complete,
        conversation_complete=True
    )
    print(f"Status: {report['data_quality_status']}")
    print(f"Confidence: {report['confidence_score']}")
    print(f"Missing Fields: {report['missing_fields']}")
    print(f"Clarification Required: {report['clarification_required']}")
    
    # Example: Missing doctor field
    print("\n" + "=" * 60)
    print("Test 2: Missing doctor field - Should need completion")
    print("=" * 60)
    
    doctor_data_incomplete: DoctorEntry = {
        "visit_id": 124,
        "patient_id": "P002",
        "disease_name": "",  # Missing!
        "medicine_prescribed": "Metformin 500mg",
        "visit_date": "2026-01-08",
        "visit_type": "new",
        "dosage": None,
        "frequency": None,
        "duration": None,
        "diagnosis_notes": None
    }
    
    report2 = analyze_visit_quality(
        visit_id=124,
        doctor_data=doctor_data_incomplete,
        patient_responses=patient_responses_complete,
        conversation_complete=True
    )
    print(f"Status: {report2['data_quality_status']}")
    print(f"Confidence: {report2['confidence_score']}")
    print(f"Missing Fields: {report2['missing_fields']}")
    
    # Example: Inconsistent responses
    print("\n" + "=" * 60)
    print("Test 3: Inconsistent responses - Should flag conflict")
    print("=" * 60)
    
    patient_responses_inconsistent: List[PatientResponse] = [
        {"response_id": 1, "visit_id": 125, "question_id": "Q1_language", "answer": "english", "timestamp": "2026-01-08T10:00:00"},
        {"response_id": 2, "visit_id": 125, "question_id": "Q2_medicine_started", "answer": "no", "timestamp": "2026-01-08T10:01:00"},  # Says NO to medicine
        {"response_id": 3, "visit_id": 125, "question_id": "Q3_adherence", "answer": "daily", "timestamp": "2026-01-08T10:02:00"},  # But answers adherence!
        {"response_id": 4, "visit_id": 125, "question_id": "Q5_new_symptoms", "answer": "no", "timestamp": "2026-01-08T10:04:00"},
        {"response_id": 5, "visit_id": 125, "question_id": "Q9_safety_check", "answer": "confirmed", "timestamp": "2026-01-08T10:05:00"}
    ]
    
    doctor_data_complete["visit_id"] = 125
    
    report3 = analyze_visit_quality(
        visit_id=125,
        doctor_data=doctor_data_complete,
        patient_responses=patient_responses_inconsistent,
        conversation_complete=True
    )
    print(f"Status: {report3['data_quality_status']}")
    print(f"Confidence: {report3['confidence_score']}")
    print(f"Inconsistencies: {report3['inconsistent_fields']}")
    print(f"Clarification IDs: {report3['clarification_question_ids']}")
    
    # Example: Abandoned conversation
    print("\n" + "=" * 60)
    print("Test 4: Abandoned conversation - Should be LOW_CONFIDENCE")
    print("=" * 60)
    
    partial_responses: List[PatientResponse] = [
        {"response_id": 1, "visit_id": 126, "question_id": "Q1_language", "answer": "hindi", "timestamp": "2026-01-08T10:00:00"},
        {"response_id": 2, "visit_id": 126, "question_id": "Q2_medicine_started", "answer": "yes", "timestamp": "2026-01-08T10:01:00"}
        # Conversation abandoned here
    ]
    
    doctor_data_complete["visit_id"] = 126
    
    report4 = analyze_visit_quality(
        visit_id=126,
        doctor_data=doctor_data_complete,
        patient_responses=partial_responses,
        conversation_complete=False,
        conversation_abandoned=True
    )
    print(f"Status: {report4['data_quality_status']}")
    print(f"Confidence: {report4['confidence_score']}")
    print(f"Missing Fields: {report4['missing_fields']}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
