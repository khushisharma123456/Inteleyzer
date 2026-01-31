"""
LLM Service for PV Agent
========================
Privacy-safe integration with Google Gemini API.
Generates questions, validates responses, and maps data to columns.
"""

import os
import json
from typing import Dict, Any, List, Optional
from .privacy_utils import PIIFilter

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")


class PrivacySafeLLMService:
    """
    LLM service that NEVER receives PII.
    Uses Google Gemini API for question generation and response validation.
    """
    
    # LLM Prompt for generating questions
    QUESTION_PROMPT = """
You are a Pharmacovigilance data quality assistant.

CONTEXT (NO PII - Patient identity completely hidden):
- Drug/Medication: {drug_name}
- Current Case Score: {case_score}
- Strength Level: {strength_level}
- Completeness: {completeness_percent}%
- Filled Data: {filled_columns}
- Missing Data: {missing_columns}
- Symptoms Reported: {symptoms}
- Risk Level: {risk_level}

PREVIOUS RESPONSES (if any):
{previous_responses}

TASK:
1. Analyze what data is STILL missing to improve case quality
2. Suggest 2-3 specific questions to ask the patient
3. For each question, indicate which database column it maps to

AVAILABLE COLUMNS TO MAP TO:
- symptom_onset_date (when symptoms started)
- symptom_resolution_date (when symptoms ended)
- doctor_confirmed (was a doctor consulted)
- hospital_confirmed (hospital records exist)
- symptoms (more symptom details)
- risk_level (severity assessment)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "analysis": "Brief explanation of data gaps",
  "suggested_questions": [
    {{"question": "When did you first notice these symptoms?", "maps_to_column": "symptom_onset_date"}},
    {{"question": "Have you consulted a doctor about this?", "maps_to_column": "doctor_confirmed"}}
  ],
  "priority": "high"
}}

RULES:
- Questions must be simple and patient-friendly
- Only map to EXISTING columns listed above
- Focus on most critical missing data first
"""

    # LLM Prompt for validating responses
    VALIDATION_PROMPT = """
You are a Pharmacovigilance data validator.

QUESTION ASKED: {question}
MAPPED TO COLUMN: {column}
PATIENT RESPONSE: {response}

TASK:
Validate if the response is useful and extract structured data.

OUTPUT FORMAT (JSON only):
{{
  "is_useful": true/false,
  "extracted_value": "the structured value to store",
  "column": "{column}",
  "confidence": "high/medium/low",
  "reason": "why this is/isn't useful"
}}

For date questions, extract dates in YYYY-MM-DD format if possible.
For yes/no questions, return true/false.
For symptom details, summarize the key information.
"""

    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_API_KEY')
        self.model = None
        
        if GENAI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Try different model names - gemini-1.5-flash or gemini-pro
                try:
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                except:
                    try:
                        self.model = genai.GenerativeModel('gemini-pro')
                    except:
                        self.model = genai.GenerativeModel('gemini-1.0-pro')
                print("✅ LLM Service initialized with Gemini API")
            except Exception as e:
                print(f"⚠️ LLM Service init error: {e}")
    
    def is_configured(self) -> bool:
        """Check if LLM is properly configured."""
        return self.model is not None
    
    def get_missing_field_questions(self, patient, previous_responses: Dict = None) -> Dict[str, Any]:
        """
        Ask LLM for questions to improve data completeness.
        
        Args:
            patient: Patient model (PII will be filtered)
            previous_responses: Dict of previous day's responses
            
        Returns:
            Dict with suggested_questions and analysis
        """
        # Filter PII - CRITICAL
        safe_data = PIIFilter.prepare_for_llm(patient)
        
        # Prepare prompt
        prompt = self.QUESTION_PROMPT.format(
            drug_name=safe_data.get('drug_name', 'Unknown'),
            case_score=safe_data.get('case_score', 0),
            strength_level=safe_data.get('strength_level', 'Unknown'),
            completeness_percent=safe_data.get('completeness_percent', 0),
            filled_columns=', '.join(safe_data.get('filled_columns', [])),
            missing_columns=', '.join(safe_data.get('missing_columns', [])),
            symptoms=safe_data.get('symptoms', 'Not specified'),
            risk_level=safe_data.get('risk_level', 'Unknown'),
            previous_responses=json.dumps(previous_responses or {}, indent=2)
        )
        
        # Call LLM
        if not self.is_configured():
            # Fallback: return predefined questions
            return self._get_fallback_questions(safe_data.get('missing_columns', []))
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Parse JSON from response
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            return json.loads(result_text)
            
        except Exception as e:
            print(f"⚠️ LLM question generation error: {e}")
            return self._get_fallback_questions(safe_data.get('missing_columns', []))
    
    def validate_response(self, question: str, column: str, response: str) -> Dict[str, Any]:
        """
        Ask LLM to validate if patient response is useful.
        
        Args:
            question: The question that was asked
            column: Database column it maps to
            response: Patient's response text
            
        Returns:
            Dict with is_useful, extracted_value, confidence
        """
        if not self.is_configured():
            return self._fallback_validation(column, response)
        
        prompt = self.VALIDATION_PROMPT.format(
            question=question,
            column=column,
            response=response
        )
        
        try:
            llm_response = self.model.generate_content(prompt)
            result_text = llm_response.text.strip()
            
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            return json.loads(result_text)
            
        except Exception as e:
            print(f"⚠️ LLM validation error: {e}")
            return self._fallback_validation(column, response)
    
    def detect_patient_intent(self, response_text: str) -> str:
        """
        Detect if patient says they are "fine" or "not fine".
        
        Returns:
            "fine" - patient is okay, stop follow-ups
            "not_fine" - patient still has issues, continue
            "unclear" - can't determine
        """
        response_lower = response_text.lower()
        
        fine_keywords = ['fine', 'okay', 'ok', 'better', 'recovered', 'good', 'well', 'no issues']
        not_fine_keywords = ['not fine', 'still', 'worse', 'bad', 'pain', 'problem', 'issue']
        
        # Check not_fine first (more specific)
        for keyword in not_fine_keywords:
            if keyword in response_lower:
                return 'not_fine'
        
        for keyword in fine_keywords:
            if keyword in response_lower:
                return 'fine'
        
        return 'unclear'
    
    def _get_fallback_questions(self, missing_columns: List[str]) -> Dict[str, Any]:
        """Fallback questions when LLM is not available."""
        predefined = {
            'symptom_onset_date': {
                'question': 'When did you first start experiencing these symptoms?',
                'maps_to_column': 'symptom_onset_date'
            },
            'symptom_resolution_date': {
                'question': 'Have your symptoms improved or resolved? If so, when?',
                'maps_to_column': 'symptom_resolution_date'
            },
            'doctor_confirmed': {
                'question': 'Have you consulted a doctor about these symptoms?',
                'maps_to_column': 'doctor_confirmed'
            },
            'hospital_confirmed': {
                'question': 'Did you visit a hospital or clinic for this issue?',
                'maps_to_column': 'hospital_confirmed'
            },
            'symptoms': {
                'question': 'Can you describe your current symptoms in more detail?',
                'maps_to_column': 'symptoms'
            }
        }
        
        questions = []
        for col in missing_columns[:3]:  # Max 3 questions
            if col in predefined:
                questions.append(predefined[col])
        
        return {
            'analysis': 'Using predefined questions (LLM not configured)',
            'suggested_questions': questions,
            'priority': 'medium'
        }
    
    def _fallback_validation(self, column: str, response: str) -> Dict[str, Any]:
        """Fallback validation when LLM is not available."""
        # Simple rule-based validation
        is_useful = len(response.strip()) > 2
        
        extracted = response.strip()
        
        # Basic extraction for known types
        if column in ['doctor_confirmed', 'hospital_confirmed']:
            response_lower = response.lower()
            if 'yes' in response_lower:
                extracted = True
            elif 'no' in response_lower:
                extracted = False
            else:
                extracted = None
        
        return {
            'is_useful': is_useful,
            'extracted_value': extracted,
            'column': column,
            'confidence': 'low',
            'reason': 'Fallback validation (LLM not configured)'
        }

    # Prompt for extracting data from voluntary/unsolicited messages
    VOLUNTARY_MESSAGE_PROMPT = """
You are a Pharmacovigilance data extraction assistant.

A patient has sent a voluntary message (not in response to a question).
Analyze the message and extract any useful health/medical data.

PATIENT MESSAGE: {message}

CURRENT PATIENT CONTEXT:
- Drug: {drug_name}
- Previous Symptoms: {symptoms}

AVAILABLE DATABASE COLUMNS:
- symptoms (text: description of symptoms, side effects)
- symptom_onset_date (date: when symptoms started, format YYYY-MM-DD)
- symptom_resolution_date (date: when symptoms ended, format YYYY-MM-DD)
- doctor_confirmed (boolean: whether patient consulted a doctor)
- hospital_confirmed (boolean: whether patient visited hospital)
- risk_level (enum: Low/Medium/High/Critical)

TASK:
1. Extract ALL relevant medical data from the message
2. Determine if patient is currently suffering or has recovered
3. Map data to appropriate columns

OUTPUT FORMAT (JSON only, no markdown):
{{
  "patient_status": "suffering" or "recovered" or "unclear",
  "is_health_related": true/false,
  "extracted_data": [
    {{"column": "symptoms", "value": "headache and nausea", "confidence": "high"}},
    {{"column": "risk_level", "value": "Medium", "confidence": "medium"}}
  ],
  "should_start_followup": true/false,
  "summary": "Brief summary of what patient reported"
}}

RULES:
- If patient says they are "fine", "okay", "recovered", "better now" → patient_status = "recovered", should_start_followup = false
- If patient describes ongoing symptoms → patient_status = "suffering", should_start_followup = true
- Extract dates in YYYY-MM-DD format when possible
- For unclear messages, set is_health_related = false
"""

    def extract_from_voluntary_message(self, message: str, patient) -> Dict[str, Any]:
        """
        Extract data from a voluntary/unsolicited patient message using LLM.
        
        Args:
            message: The voluntary message from patient
            patient: Patient model object for context
            
        Returns:
            Dict with extracted_data, patient_status, should_start_followup
        """
        if not self.is_configured():
            return self._fallback_voluntary_extraction(message, patient)
        
        try:
            from .privacy_utils import PIIFilter
            
            prompt = self.VOLUNTARY_MESSAGE_PROMPT.format(
                message=message,
                drug_name=patient.drug_name if patient else 'Unknown',
                symptoms=patient.symptoms if patient else 'None reported'
            )
            
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Parse JSON
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            import json
            result = json.loads(result_text)
            return result
            
        except Exception as e:
            print(f"⚠️ LLM voluntary extraction error: {e}")
            return self._fallback_voluntary_extraction(message, patient)
    
    def _fallback_voluntary_extraction(self, message: str, patient) -> Dict[str, Any]:
        """Fallback extraction when LLM is not available."""
        message_lower = message.lower()
        
        # Check if patient is recovered
        recovery_words = ['fine', 'okay', 'ok', 'better', 'recovered', 'cured', 'well now', 'no problem', 'good now']
        is_recovered = any(word in message_lower for word in recovery_words)
        
        # Check if patient is suffering
        suffering_words = ['pain', 'suffering', 'problem', 'issue', 'symptom', 'side effect', 'headache', 
                          'nausea', 'vomit', 'dizziness', 'rash', 'fever', 'sick', 'worse', 'bad']
        is_suffering = any(word in message_lower for word in suffering_words)
        
        # Determine status
        if is_recovered and not is_suffering:
            status = 'recovered'
            should_followup = False
        elif is_suffering:
            status = 'suffering'
            should_followup = True
        else:
            status = 'unclear'
            should_followup = False
        
        # Basic data extraction
        extracted_data = []
        
        # Always save the message as symptoms
        if is_suffering or len(message) > 10:
            extracted_data.append({
                'column': 'symptoms',
                'value': message,
                'confidence': 'medium'
            })
        
        # Check for severity indicators
        if any(word in message_lower for word in ['severe', 'critical', 'emergency', 'hospital', 'icu']):
            extracted_data.append({
                'column': 'risk_level',
                'value': 'Critical',
                'confidence': 'high'
            })
        elif any(word in message_lower for word in ['bad', 'serious', 'worried']):
            extracted_data.append({
                'column': 'risk_level',
                'value': 'High',
                'confidence': 'medium'
            })
        
        # Check for doctor/hospital mentions
        if any(word in message_lower for word in ['doctor', 'physician', 'clinic', 'consulted dr']):
            extracted_data.append({
                'column': 'doctor_confirmed',
                'value': True,
                'confidence': 'medium'
            })
        
        if any(word in message_lower for word in ['hospital', 'admitted', 'emergency room', 'er visit']):
            extracted_data.append({
                'column': 'hospital_confirmed',
                'value': True,
                'confidence': 'high'
            })
        
        return {
            'patient_status': status,
            'is_health_related': is_suffering or is_recovered,
            'extracted_data': extracted_data,
            'should_start_followup': should_followup,
            'summary': f"Patient message: {message[:100]}..."
        }


# Predefined questions that are always included
PREDEFINED_QUESTIONS = [
    {
        'question': 'How are you feeling today?',
        'maps_to_column': None,  # Used for intent detection
        'purpose': 'wellness_check'
    },
    {
        'question': 'Are you still experiencing any symptoms from the medication?',
        'maps_to_column': 'symptoms',
        'purpose': 'symptom_update'
    }
]


def get_combined_questions(patient, previous_responses: Dict = None) -> Dict[str, Any]:
    """
    Get combined predefined + LLM questions for a patient.
    
    Returns:
        Dict with predefined_questions, llm_questions, and all_questions
    """
    service = PrivacySafeLLMService()
    llm_result = service.get_missing_field_questions(patient, previous_responses)
    
    return {
        'predefined_questions': PREDEFINED_QUESTIONS,
        'llm_questions': llm_result.get('suggested_questions', []),
        'all_questions': PREDEFINED_QUESTIONS + llm_result.get('suggested_questions', []),
        'analysis': llm_result.get('analysis', ''),
        'priority': llm_result.get('priority', 'medium')
    }
