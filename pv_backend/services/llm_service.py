"""
LLM Service for PV Agent
========================
Privacy-safe integration with Google Gemini API, Groq API, or OpenAI API.
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

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ openai not installed. Run: pip install openai")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ groq not installed. Run: pip install groq")


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

    # ============================================================================
    # DAY-SPECIFIC PERSONALIZED QUESTION PROMPT (Using Case Scoring for Gemini)
    # ============================================================================
    DAY_SPECIFIC_QUESTION_PROMPT = """You are a Pharmacovigilance follow-up specialist. Generate 2-3 UNIQUE personalized questions.

CONTEXT:
- Day {current_day} of 7-day follow-up
- Drug: {drug_name}
- Symptoms: {symptoms}
- Case Score: {case_score}, Strength: {strength_level}
- Missing data: {missing_columns}
- User's preferred language: {language}

DAY FOCUS: {day_focus}

PREDEFINED QUESTIONS ALREADY BEING ASKED (DO NOT REPEAT THESE):
{predefined_questions_text}

YOUR TASK: Generate 2-3 DIFFERENT questions that:
1. Are NOT similar to the predefined questions above
2. Focus on aspects NOT covered by predefined questions
3. Are personalized based on the drug ({drug_name}) and symptoms
4. Help gather additional pharmacovigilance data

SUGGESTED UNIQUE QUESTION TOPICS BY DAY:
- Day 1: Medication timing, food interactions, exact symptom description, allergies
- Day 3: Daily activities impact, sleep quality, appetite changes, other medications taken
- Day 5: Work/school impact, family support, treatment effectiveness, mood changes
- Day 7: Overall experience summary, likelihood to continue medication, suggestions

IMPORTANT LANGUAGE INSTRUCTIONS:
- First write the question in ENGLISH
- Then translate it naturally to {language}
- For Hindi use proper Devanagari: क्या, हाँ, नहीं, आप, कैसे, क्यों
- For Telugu use proper script: మీరు, హాయ్, ఎలా
- Options should be simple words in {language}

EXAMPLES OF GOOD TRANSLATIONS:
- English: "Have you experienced any side effects?" -> Hindi: "क्या आपको कोई साइड इफेक्ट्स हुए हैं?"
- English: "Yes" -> Hindi: "हाँ", Telugu: "అవును"  
- English: "No" -> Hindi: "नहीं", Telugu: "కాదు"
- English: "Not sure" -> Hindi: "पता नहीं", Telugu: "తెలియదు"

OUTPUT JSON (no markdown, proper escaping):
{{
  "analysis": "Brief reason for questions - explain how these differ from predefined",
  "suggested_questions": [
    {{
      "id": "llm_day{current_day}_q1",
      "question": "Question in {language} with proper script",
      "question_english": "Same question in English",
      "maps_to_column": "symptoms",
      "purpose": "purpose - different from predefined",
      "options": [
        {{"key": "yes", "text": "Yes in {language}", "text_english": "Yes"}},
        {{"key": "no", "text": "No in {language}", "text_english": "No"}},
        {{"key": "unsure", "text": "Not sure in {language}", "text_english": "Not sure"}}
      ]
    }}
  ],
  "priority": "high"
}}

RULES:
- DO NOT repeat or rephrase any predefined questions
- Write questions in {language} with proper native script
- Provide 3-5 simple answer options per question
- Keep JSON valid - do not escape question marks or special characters
- Ask about: side effects details, lifestyle impact, medication experience, emotional wellbeing"""

    # Day-specific focus areas for the LLM prompt
    DAY_FOCUS_AREAS = {
        1: """Day 1 Focus: Initial assessment
- Confirm current symptom status
- Establish symptom timeline/onset
- Assess initial severity
- Build rapport with patient""",
        
        3: """Day 3 Focus: Symptom progression
- Track how symptoms have changed
- Check if medical help was sought
- Assess medication compliance
- Identify any new symptoms""",
        
        5: """Day 5 Focus: Clinical impact
- Evaluate symptom trend (improving/worsening)
- Document hospital visits if any
- Understand daily life impact
- Record any treatments taken""",
        
        7: """Day 7 Focus: Resolution and closure
- Final symptom status
- Document resolution date if applicable
- Gather final feedback
- Offer follow-up care options"""
    }

    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_API_KEY')
        self.openai_key = os.environ.get('OPENAI_API_KEY')
        self.groq_key = os.environ.get('GROQ_API_KEY')
        self.model = None
        self.openai_client = None
        self.groq_client = None
        self.llm_provider = None  # 'gemini', 'openai', or 'groq'
        self._init_attempted = False
        
        # Try Gemini first
        if GENAI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Try different model names in order of preference
                model_names = [
                    'gemini-2.0-flash-lite',  # Lightweight model
                    'gemini-2.0-flash',       # Latest flash model
                    'gemini-1.5-pro',         # Pro model
                    'gemini-pro',             # Legacy pro
                ]
                
                for model_name in model_names:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        # Test the model with a simple call
                        test_response = self.model.generate_content("Say 'OK'")
                        print(f"✅ LLM Service initialized with Gemini {model_name}")
                        self.llm_provider = 'gemini'
                        self._init_attempted = True
                        break
                    except Exception as model_error:
                        error_str = str(model_error)
                        if '429' in error_str or 'quota' in error_str.lower():
                            print(f"⚠️ Gemini quota exhausted - trying Groq...")
                            self.model = None
                            break
                        else:
                            self.model = None
                            continue
                        
            except Exception as e:
                print(f"⚠️ Gemini init error: {e}")
        
        # Try Groq as second option (fast and free tier available)
        if not self.model and GROQ_AVAILABLE and self.groq_key:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
                # Test with a simple call
                test_response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Say 'OK'"}],
                    max_tokens=10
                )
                print(f"✅ LLM Service initialized with Groq (Llama 3.1)")
                self.llm_provider = 'groq'
                self._init_attempted = True
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                    print(f"⚠️ Groq rate limited - trying OpenAI...")
                else:
                    print(f"⚠️ Groq init error: {e} - trying OpenAI...")
                self.groq_client = None
        
        # Try OpenAI as fallback
        if not self.model and not self.groq_client and OPENAI_AVAILABLE and self.openai_key:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_key)
                # Test with a simple call
                test_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say 'OK'"}],
                    max_tokens=10
                )
                print(f"✅ LLM Service initialized with OpenAI GPT-3.5-turbo")
                self.llm_provider = 'openai'
                self._init_attempted = True
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                    print(f"⚠️ OpenAI quota exhausted - using fallback questions")
                else:
                    print(f"⚠️ OpenAI init error: {e} - using fallback questions")
                self.openai_client = None
        
        if not self.model and not self.groq_client and not self.openai_client:
            if not self._init_attempted:
                print("⚠️ No LLM available - using fallback questions")
            self._init_attempted = True
    
    def is_configured(self) -> bool:
        """Check if LLM is properly configured."""
        return self.model is not None or self.groq_client is not None or self.openai_client is not None
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM (Gemini, Groq, or OpenAI) with the prompt."""
        if self.llm_provider == 'gemini' and self.model:
            response = self.model.generate_content(prompt)
            return response.text
        elif self.llm_provider == 'groq' and self.groq_client:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.5
            )
            return response.choices[0].message.content
        elif self.llm_provider == 'openai' and self.openai_client:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            raise Exception("No LLM configured")
    
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
            result_text = self._call_llm(prompt).strip()
            
            # Parse JSON from response
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            return json.loads(result_text)
            
        except Exception as e:
            print(f"⚠️ LLM question generation error: {e}")
            return self._get_fallback_questions(safe_data.get('missing_columns', []))
    
    def get_personalized_day_questions(self, patient, previous_responses: Dict = None, current_day: int = 1, language: str = "English") -> Dict[str, Any]:
        """
        Generate personalized questions for a specific day using case scoring data.
        This method sends case scoring information to Gemini/Groq to generate 
        day-appropriate personalized questions in the patient's preferred language.
        
        Args:
            patient: Patient model (PII will be filtered)
            previous_responses: Dict of previous day's responses
            current_day: The current follow-up day (1, 3, 5, or 7)
            language: Patient's preferred language (e.g., 'Telugu', 'Hindi', 'English')
            
        Returns:
            Dict with suggested_questions, analysis, and focus_areas
        """
        # Filter PII - CRITICAL
        safe_data = PIIFilter.prepare_for_llm(patient)
        
        # Get day focus area
        day_focus = self.DAY_FOCUS_AREAS.get(current_day, self.DAY_FOCUS_AREAS[1])
        
        # Get predefined questions for this day to tell LLM what NOT to ask
        predefined = get_day_specific_questions(current_day)
        predefined_text = "\n".join([f"- {q.get('question', '')}" for q in predefined])
        
        # Prepare prompt with case scoring data for personalization
        prompt = self.DAY_SPECIFIC_QUESTION_PROMPT.format(
            current_day=current_day,
            drug_name=safe_data.get('drug_name', 'Unknown medication'),
            symptoms=safe_data.get('symptoms', 'Not specified'),
            case_score=safe_data.get('case_score', 0),
            strength_level=safe_data.get('strength_level', 'Unknown'),
            missing_columns=', '.join(safe_data.get('missing_columns', [])),
            day_focus=day_focus,
            language=language,
            predefined_questions_text=predefined_text
        )
        
        # Call LLM
        if not self.is_configured():
            # Fallback: return day-specific fallback questions
            return self._get_day_fallback_questions(current_day, safe_data.get('missing_columns', []))
        
        try:
            print(f"🤖 Calling {self.llm_provider} to generate UNIQUE questions in {language} for Day {current_day}...")
            print(f"📋 Predefined questions to avoid: {len(predefined)}")
            result_text = self._call_llm(prompt).strip()
            
            # Log raw LLM response
            print(f"📝 Raw LLM Response:\n{result_text}")
            
            # Parse JSON from response - handle markdown code blocks
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            # Try to fix common JSON issues
            result_text = result_text.strip()
            
            # Fix invalid escape sequences that LLM sometimes generates
            # e.g., \? \! \. should just be ? ! .
            import re
            result_text = re.sub(r'\\([?!.,;:\'\"])', r'\1', result_text)
            
            result = json.loads(result_text)
            
            # Ensure questions have proper IDs and mark as LLM-generated
            for i, q in enumerate(result.get('suggested_questions', [])):
                if 'id' not in q:
                    q['id'] = f'llm_day{current_day}_q{i+1}'
                q['source'] = 'llm'
                q['llm_provider'] = self.llm_provider
                q['language'] = language
            
            print(f"✅ Generated {len(result.get('suggested_questions', []))} personalized questions for Day {current_day} in {language}")
            print(f"📋 LLM Questions: {json.dumps(result.get('suggested_questions', []), ensure_ascii=False, indent=2)}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON Parse Error: {e}")
            print(f"📝 Full response that failed to parse:\n{result_text}")
            return self._get_day_fallback_questions(current_day, safe_data.get('missing_columns', []))
        except Exception as e:
            print(f"⚠️ LLM day-specific question generation error: {e}")
            return self._get_day_fallback_questions(current_day, safe_data.get('missing_columns', []))
    
    def _get_day_fallback_questions(self, current_day: int, missing_columns: List[str]) -> Dict[str, Any]:
        """Fallback questions for specific day when LLM is not available."""
        
        # Day-specific fallback questions based on focus areas
        day_fallbacks = {
            1: [
                {'id': 'fb_day1_q1', 'question': 'Can you describe your symptoms in more detail?', 'maps_to_column': 'symptoms', 'purpose': 'symptom_detail'},
                {'id': 'fb_day1_q2', 'question': 'Are you experiencing any discomfort right now?', 'maps_to_column': 'symptoms', 'purpose': 'current_status'}
            ],
            3: [
                {'id': 'fb_day3_q1', 'question': 'Have your symptoms improved since you first reported them?', 'maps_to_column': 'symptoms', 'purpose': 'progression'},
                {'id': 'fb_day3_q2', 'question': 'Did you need to take any action to manage your symptoms?', 'maps_to_column': None, 'purpose': 'actions_taken'}
            ],
            5: [
                {'id': 'fb_day5_q1', 'question': 'How are you managing with these symptoms on a daily basis?', 'maps_to_column': None, 'purpose': 'daily_impact'},
                {'id': 'fb_day5_q2', 'question': 'Have you noticed any patterns in when symptoms occur?', 'maps_to_column': 'symptoms', 'purpose': 'pattern_detection'}
            ],
            7: [
                {'id': 'fb_day7_q1', 'question': 'Looking back over the past week, how has your condition changed?', 'maps_to_column': 'symptoms', 'purpose': 'weekly_summary'},
                {'id': 'fb_day7_q2', 'question': 'Is there anything you wish we had asked about earlier?', 'maps_to_column': None, 'purpose': 'feedback'}
            ]
        }
        
        questions = day_fallbacks.get(current_day, day_fallbacks[1])
        
        # Add missing column questions if applicable
        column_questions = {
            'symptom_onset_date': {'id': f'fb_day{current_day}_onset', 'question': 'When did you first notice these symptoms?', 'maps_to_column': 'symptom_onset_date', 'purpose': 'onset_date'},
            'doctor_confirmed': {'id': f'fb_day{current_day}_doctor', 'question': 'Have you seen a doctor about this?', 'maps_to_column': 'doctor_confirmed', 'purpose': 'medical_confirmation'},
            'hospital_confirmed': {'id': f'fb_day{current_day}_hospital', 'question': 'Did you need to visit a hospital?', 'maps_to_column': 'hospital_confirmed', 'purpose': 'hospital_visit'}
        }
        
        for col in missing_columns[:1]:  # Add at most 1 additional question
            if col in column_questions:
                questions.append(column_questions[col])
        
        return {
            'analysis': f'Using fallback questions for Day {current_day} (LLM not configured)',
            'focus_areas': [self.DAY_FOCUS_AREAS.get(current_day, 'General follow-up')],
            'suggested_questions': questions,
            'priority': 'medium'
        }
    
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
            result_text = self._call_llm(prompt).strip()
            
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
        Detect if patient EXPLICITLY says they are completely fine and want to stop follow-ups.
        
        This should only return 'fine' when the patient clearly indicates they have recovered
        or don't need further follow-up. It should NOT trigger on regular answers to questions
        that happen to contain words like "okay" or "good".
        
        Returns:
            "fine" - patient explicitly says recovered, stop follow-ups
            "not_fine" - patient still has issues, continue
            "unclear" - regular answer, continue with questions
        """
        response_lower = response_text.lower().strip()
        
        # Check for NOT_FINE first (more specific - patient is suffering)
        not_fine_keywords = [
            'not fine', 'not okay', 'not good', 'not well',
            'still have', 'still suffering', 'still experiencing',
            'worse', 'getting worse', 'worsening',
            'bad', 'terrible', 'awful', 'horrible',
            'pain', 'painful', 'hurting', 'hurt',
            'problem', 'problems', 'issue', 'issues',
            'side effect', 'side effects', 'adverse',
            'need help', 'need doctor', 'need hospital',
            'vomiting', 'nausea', 'fever', 'rash', 'allergic'
        ]
        
        for keyword in not_fine_keywords:
            if keyword in response_lower:
                return 'not_fine'
        
        # Explicit FINE patterns - patient clearly says they are recovered
        # These are very specific phrases that indicate "stop following up"
        explicit_fine_phrases = [
            'i am fine now',
            'i am completely fine',
            'i am totally fine',
            'i have recovered',
            'i have fully recovered',
            'i am recovered',
            'fully recovered',
            'completely recovered',
            'no more issues',
            'no more problems',
            'no issues at all',
            'all good now',
            'feeling much better now',
            'i am all better',
            'i am cured',
            'no need for follow up',
            'no need for followup',
            'please stop',
            'stop messaging',
            'stop following up',
            'do not contact',
            'don\'t contact'
        ]
        
        for phrase in explicit_fine_phrases:
            if phrase in response_lower:
                return 'fine'
        
        # Short explicit fine responses (only if the entire message is just this)
        short_fine_responses = ['fine', 'i am fine', 'im fine', "i'm fine", 'all fine', 'all ok', 'all okay']
        if response_lower in short_fine_responses:
            return 'fine'
        
        # For anything else (including "I am feeling okay", "doing good", etc.)
        # treat as a regular answer - continue with questions
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
        from datetime import datetime, date
        import re
        
        # Simple rule-based validation
        is_useful = len(response.strip()) > 2
        
        extracted = response.strip()
        
        # Handle date columns - try to parse or skip
        if column in ['symptom_onset_date', 'symptom_resolution_date']:
            # Try to extract date from response
            date_patterns = [
                r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD/MM/YYYY or DD-MM-YYYY
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD
                r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})',  # 15 Jan 2024
            ]
            
            # Look for relative time phrases
            response_lower = response.lower()
            if any(phrase in response_lower for phrase in ['yesterday', 'today', 'last week', 'week ago', 'days ago', 'days back']):
                # Don't try to set date - mark as useful text but don't extract value for DB
                return {
                    'is_useful': True,
                    'extracted_value': None,  # Don't set - invalid date format
                    'column': column,
                    'confidence': 'low',
                    'reason': 'Contains relative time reference - storing as text only'
                }
            
            # Try to parse actual dates
            for pattern in date_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        # Attempt basic parsing
                        groups = match.groups()
                        if len(groups[0]) == 4:  # YYYY first
                            parsed = date(int(groups[0]), int(groups[1]), int(groups[2]))
                        else:  # DD first
                            parsed = date(int(groups[2]), int(groups[1]), int(groups[0]))
                        return {
                            'is_useful': True,
                            'extracted_value': parsed,
                            'column': column,
                            'confidence': 'medium',
                            'reason': 'Parsed date from response'
                        }
                    except:
                        pass
            
            # Couldn't parse date - don't set the column
            return {
                'is_useful': True,
                'extracted_value': None,
                'column': column,
                'confidence': 'low',
                'reason': 'Could not parse date from response'
            }
        
        # Handle boolean columns
        if column in ['doctor_confirmed', 'hospital_confirmed']:
            response_lower = response.lower()
            if any(word in response_lower for word in ['yes', 'consulted', 'visited', 'went']):
                extracted = True
            elif any(word in response_lower for word in ['no', 'not', 'haven\'t', 'havent', 'didn\'t', 'didnt']):
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
            
            result_text = self._call_llm(prompt).strip()
            
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

    def analyze_response_tone_and_relevance(self, question: str, user_response: str, expected_column: str) -> Dict[str, Any]:
        """
        Use Groq API to analyze tone and relevance of patient response.
        
        CRITICAL: This determines if a response is:
        1. TONE-RELEVANT: Patient is engaged and answering properly
        2. TONE-DISMISSIVE: Patient says "ok", "yes", without substance
        3. TONE-SUFFERING: Patient is in distress or concerned
        4. TONE-IRRELEVANT: Response doesn't match the question asked
        
        Args:
            question: The question asked to the patient
            user_response: Patient's response text
            expected_column: Database column this question maps to
            
        Returns:
            Dict with tone, relevance_score (0-1), and recommended_action
        """
        if not self.groq_client and not self.model and not self.openai_client:
            # Fallback to simple pattern matching
            return self._simple_tone_detection(question, user_response)
        
        tone_analysis_prompt = f"""You are a patient communication analyzer for pharmacovigilance.

Analyze this patient response:

QUESTION ASKED: {question}
PATIENT RESPONSE: {user_response}
EXPECTED DATA FIELD: {expected_column}

TASK:
1. Analyze the TONE of the response (engaged, dismissive, suffering, confused)
2. Check RELEVANCE to the question (1-10 score where 10=perfect answer)
3. Determine if response contains useful DATA (yes/no/needs clarification)

RESPONSE FORMAT (JSON only):
{{
  "tone": "engaged|dismissive|suffering|confused|irrelevant",
  "tone_confidence": 0.95,
  "relevance_score": 8,
  "has_useful_data": true,
  "extracted_value": "patient said they are better",
  "action": "proceed|re_ask|clarify|skip",
  "reasoning": "Patient provided clear answer indicating improvement"
}}

TONE DEFINITIONS:
- engaged: Patient is actively answering with substance
- dismissive: Patient gives minimal response like "ok", "yes" without details
- suffering: Patient shows signs of distress or worry
- confused: Patient doesn't understand the question
- irrelevant: Response doesn't relate to question asked

ACTION DEFINITIONS:
- proceed: Accept answer and move to next question
- re_ask: Ask question again (patient didn't understand)
- clarify: Ask for more details
- skip: Skip this question due to irrelevance"""

        try:
            if self.llm_provider == 'groq' and self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": tone_analysis_prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                result_text = response.choices[0].message.content
            else:
                # Fall back to simple detection
                return self._simple_tone_detection(question, user_response)
            
            # Parse JSON response
            import json
            json_match = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_match != -1 and json_end > json_match:
                json_str = result_text[json_match:json_end]
                analysis = json.loads(json_str)
                return analysis
            else:
                return self._simple_tone_detection(question, user_response)
                
        except Exception as e:
            print(f"⚠️ Tone analysis error: {e} - using fallback")
            return self._simple_tone_detection(question, user_response)

    def _simple_tone_detection(self, question: str, response: str) -> Dict[str, Any]:
        """Fallback tone detection using pattern matching."""
        response_lower = response.lower().strip()
        
        # Dismissive patterns
        dismissive_words = ['ok', 'yes', 'no', 'fine', 'ok ok', 'yes ok', 'mmm', 'yeah', 'yep', 'nope']
        is_dismissive = len(response) < 5 and any(w == response_lower for w in dismissive_words)
        
        # Suffering patterns
        suffering_words = ['hurt', 'pain', 'bad', 'worse', 'worry', 'concern', 'problem', 'issue', 'severe', 'critical', 'scared', 'fear']
        has_suffering = any(w in response_lower for w in suffering_words)
        
        # Engagement (has adjectives, adverbs, details)
        engagement_indicators = ['because', 'since', 'when', 'if', 'but', 'however', 'although', 'while']
        is_detailed = any(w in response_lower for w in engagement_indicators) or len(response) > 20
        
        # Determine tone
        if is_dismissive:
            tone = 'dismissive'
            action = 'clarify' if has_suffering else 'skip'
            confidence = 0.8
            relevance = 3
        elif has_suffering:
            tone = 'suffering'
            action = 'clarify'
            confidence = 0.85
            relevance = 7
        elif is_detailed:
            tone = 'engaged'
            action = 'proceed'
            confidence = 0.9
            relevance = 8
        else:
            tone = 'confused'
            action = 're_ask'
            confidence = 0.6
            relevance = 4
        
        return {
            'tone': tone,
            'tone_confidence': confidence,
            'relevance_score': relevance,
            'has_useful_data': relevance >= 5,
            'action': action,
            'reasoning': f"Fallback detection: {tone} response"
        }

    def should_repeat_question(self, analysis: Dict[str, Any], times_asked: int = 1) -> bool:
        """
        Determine if a question should be asked again based on tone analysis.
        
        Args:
            analysis: Result from analyze_response_tone_and_relevance()
            times_asked: How many times this question has been asked
            
        Returns:
            bool: True if question should be re-asked, False to move forward
        """
        action = analysis.get('action', 'proceed')
        relevance = analysis.get('relevance_score', 5)
        
        # Don't repeat more than 2 times
        if times_asked >= 2:
            return False
        
        # Re-ask if dismissed or irrelevant
        if action in ['re_ask', 'clarify'] and relevance < 6:
            return True
        
        # Never repeat if person is suffering (need to support, not push)
        if analysis.get('tone') == 'suffering':
            return False
        
        return False


# ============================================================================
# DAY-SPECIFIC PREDEFINED QUESTIONS FOR 1/3/5/7 FOLLOW-UP CYCLE
# ============================================================================
# These questions are designed to progressively gather information across
# the follow-up cycle, starting with basic wellness and progressing to
# more detailed clinical data needed by pharmaceutical companies.

# Multi-language question translations
QUESTION_TRANSLATIONS = {
    'day1_q1': {
        'English': 'How are you feeling today after taking the medication?',
        'Hindi': 'दवाई लेने के बाद आज आप कैसा महसूस कर रहे हैं?',
        'Telugu': 'మందు తీసుకున్న తర్వాత మీరు ఈరోజు ఎలా ఫీల్ అవుతున్నారు?',
        'Bengali': 'ওষুধ খাওয়ার পর আজ আপনি কেমন অনুভব করছেন?',
        'Marathi': 'औषध घेतल्यानंतर आज तुम्हाला कसे वाटते?',
        'Tamil': 'மருந்து எடுத்த பிறகு இன்று நீங்கள் எப்படி உணர்கிறீர்கள்?',
        'Gujarati': 'દવા લીધા પછી આજે તમને કેવું લાગે છે?',
        'Kannada': 'ಔಷಧಿ ತೆಗೆದುಕೊಂಡ ನಂತರ ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಅನಿಸುತ್ತಿದೆ?',
        'Malayalam': 'മരുന്ന് കഴിച്ചതിന് ശേഷം ഇന്ന് നിങ്ങൾക്ക് എങ്ങനെ തോന്നുന്നു?',
        'Punjabi': 'ਦਵਾਈ ਲੈਣ ਤੋਂ ਬਾਅਦ ਅੱਜ ਤੁਸੀਂ ਕਿਵੇਂ ਮਹਿਸੂਸ ਕਰ ਰਹੇ ਹੋ?',
        'Odia': 'ଔଷଧ ଖାଇବା ପରେ ଆଜି ଆପଣ କେମିତି ଅନୁଭବ କରୁଛନ୍ତି?',
        'Assamese': 'ঔষধ খোৱাৰ পিছত আজি আপুনি কেনে অনুভৱ কৰিছে?',
        'Urdu': 'دوا لینے کے بعد آج آپ کیسا محسوس کر رہے ہیں؟'
    },
    'day1_q2': {
        'English': 'Are you still experiencing the symptoms you reported earlier?',
        'Hindi': 'क्या आप अभी भी उन लक्षणों का अनुभव कर रहे हैं जो आपने पहले बताए थे?',
        'Telugu': 'మీరు ఇంతకు ముందు చెప్పిన లక్షణాలు ఇప్పటికీ అనుభవిస్తున్నారా?',
        'Bengali': 'আপনি আগে যে লক্ষণগুলি জানিয়েছিলেন সেগুলি এখনও অনুভব করছেন?',
        'Marathi': 'तुम्ही आधी सांगितलेली लक्षणे अजूनही अनुभवत आहात का?',
        'Tamil': 'நீங்கள் முன்பு தெரிவித்த அறிகுறிகளை இன்னும் அனுபவிக்கிறீர்களா?',
        'Gujarati': 'તમે પહેલાં જણાવેલા લક્ષણો હજુ પણ અનુભવી રહ્યા છો?',
        'Kannada': 'ನೀವು ಮೊದಲು ವರದಿ ಮಾಡಿದ ರೋಗಲಕ್ಷಣಗಳು ಇನ್ನೂ ಅನುಭವಿಸುತ್ತಿದ್ದೀರಾ?',
        'Malayalam': 'നിങ്ങൾ മുമ്പ് റിപ്പോർട്ട് ചെയ്ത ലക്ഷണങ്ങൾ ഇപ്പോഴും അനുഭവിക്കുന്നുണ്ടോ?',
        'Punjabi': 'ਕੀ ਤੁਸੀਂ ਅਜੇ ਵੀ ਉਹ ਲੱਛਣ ਮਹਿਸੂਸ ਕਰ ਰਹੇ ਹੋ ਜੋ ਤੁਸੀਂ ਪਹਿਲਾਂ ਦੱਸੇ ਸਨ?',
        'Odia': 'ଆପଣ ପୂର୍ବରୁ କହିଥିବା ଲକ୍ଷଣଗୁଡ଼ିକ ଏବେ ବି ଅନୁଭବ କରୁଛନ୍ତି କି?',
        'Assamese': 'আপুনি আগতে কোৱা লক্ষণবোৰ এতিয়াও অনুভৱ কৰি আছে নেকি?',
        'Urdu': 'کیا آپ اب بھی وہ علامات محسوس کر رہے ہیں جو آپ نے پہلے بتائی تھیں؟'
    },
    'day1_q3': {
        'English': 'On a scale of 1-10, how would you rate the severity of your symptoms?',
        'Hindi': '1-10 के पैमाने पर, आप अपने लक्षणों की गंभीरता को कैसे रेट करेंगे?',
        'Telugu': '1-10 స్కేల్‌లో, మీ లక్షణాల తీవ్రతను మీరు ఎలా రేట్ చేస్తారు?',
        'Bengali': '1-10 স্কেলে, আপনি আপনার লক্ষণগুলির তীব্রতা কীভাবে রেট করবেন?',
        'Marathi': '1-10 स्केलवर, तुम्ही तुमच्या लक्षणांची तीव्रता कशी रेट कराल?',
        'Tamil': '1-10 அளவில், உங்கள் அறிகுறிகளின் தீவிரத்தை எவ்வாறு மதிப்பிடுவீர்கள்?',
        'Gujarati': '1-10 સ્કેલ પર, તમે તમારા લક્ષણોની તીવ્રતાને કેવી રીતે રેટ કરશો?',
        'Kannada': '1-10 ಪ್ರಮಾಣದಲ್ಲಿ, ನಿಮ್ಮ ರೋಗಲಕ್ಷಣಗಳ ತೀವ್ರತೆಯನ್ನು ಹೇಗೆ ರೇಟ್ ಮಾಡುತ್ತೀರಿ?',
        'Malayalam': '1-10 സ്കെയിലിൽ, നിങ്ങളുടെ ലക്ഷണങ്ങളുടെ തീവ്രത എങ്ങനെ റേറ്റ് ചെയ്യും?',
        'Punjabi': '1-10 ਦੇ ਪੈਮਾਨੇ ਤੇ, ਤੁਸੀਂ ਆਪਣੇ ਲੱਛਣਾਂ ਦੀ ਗੰਭੀਰਤਾ ਨੂੰ ਕਿਵੇਂ ਰੇਟ ਕਰੋਗੇ?',
        'Odia': '1-10 ସ୍କେଲରେ, ଆପଣ ଆପଣଙ୍କ ଲକ୍ଷଣଗୁଡ଼ିକର ଗୁରୁତ୍ୱ କେମିତି ରେଟ୍ କରିବେ?',
        'Assamese': '1-10 স্কেলত, আপুনি আপোনাৰ লক্ষণৰ তীব্ৰতা কেনেকৈ ৰেট কৰিব?',
        'Urdu': '1-10 کے پیمانے پر، آپ اپنی علامات کی شدت کو کیسے درجہ دیں گے؟'
    },
    'day1_q4': {
        'English': 'When did you first notice these symptoms?',
        'Hindi': 'आपने पहली बार ये लक्षण कब नोटिस किए?',
        'Telugu': 'మీరు ఈ లక్షణాలను మొదట ఎప్పుడు గమనించారు?',
        'Bengali': 'আপনি প্রথম কবে এই লক্ষণগুলি লক্ষ্য করেছিলেন?',
        'Marathi': 'तुम्हाला ही लक्षणे पहिल्यांदा कधी दिसली?',
        'Tamil': 'இந்த அறிகுறிகளை நீங்கள் முதலில் எப்போது கவனித்தீர்கள்?',
        'Gujarati': 'તમે આ લક્ષણો પહેલીવાર ક્યારે જોયા?',
        'Kannada': 'ನೀವು ಈ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಮೊದಲು ಯಾವಾಗ ಗಮನಿಸಿದಿರಿ?',
        'Malayalam': 'ഈ ലക്ഷണങ്ങൾ നിങ്ങൾ ആദ്യം എപ്പോൾ ശ്രദ്ധിച്ചു?',
        'Punjabi': 'ਤੁਸੀਂ ਇਹ ਲੱਛਣ ਪਹਿਲੀ ਵਾਰ ਕਦੋਂ ਦੇਖੇ?',
        'Odia': 'ଏହି ଲକ୍ଷଣଗୁଡ଼ିକ ଆପଣ ପ୍ରଥମେ କେବେ ଲକ୍ଷ୍ୟ କରିଥିଲେ?',
        'Assamese': 'এই লক্ষণবোৰ আপুনি প্ৰথমে কেতিয়া দেখিছিল?',
        'Urdu': 'آپ نے یہ علامات پہلی بار کب محسوس کیں؟'
    },
    # Day 3 Questions
    'day3_q1': {
        'English': 'How have your symptoms changed since we last spoke?',
        'Hindi': 'पिछली बार बात करने के बाद आपके लक्षणों में क्या बदलाव आया?',
        'Telugu': 'మనం చివరిసారి మాట్లాడినప్పటి నుండి మీ లక్షణాలు ఎలా మారాయి?',
        'Bengali': 'শেষ কথা বলার পর থেকে আপনার লক্ষণ কীভাবে বদলেছে?',
        'Marathi': 'शेवटच्या संभाषणानंतर तुमच्या लक्षणांमध्ये काय बदल झाला?',
        'Tamil': 'கடைசியாக பேசியதிலிருந்து உங்கள் அறிகுறிகள் எவ்வாறு மாறின?',
        'Gujarati': 'છેલ્લી વખત વાત કર્યા પછી તમારા લક્ષણોમાં શું બદલાવ આવ્યો?',
        'Kannada': 'ಕೊನೆಯ ಬಾರಿ ಮಾತನಾಡಿದ ನಂತರ ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಹೇಗೆ ಬದಲಾದವು?',
        'Malayalam': 'കഴിഞ്ഞ തവണ സംസാരിച്ചതിന് ശേഷം നിങ്ങളുടെ ലക്ഷണങ്ങൾ എങ്ങനെ മാറി?',
        'Punjabi': 'ਆਖਰੀ ਵਾਰ ਗੱਲ ਕਰਨ ਤੋਂ ਬਾਅਦ ਤੁਹਾਡੇ ਲੱਛਣ ਕਿਵੇਂ ਬਦਲੇ?',
        'Odia': 'ଶେଷ ଥର କଥା ହେବା ପରେ ଆପଣଙ୍କ ଲକ୍ଷଣରେ କଣ ବଦଳ ଆସିଛି?',
        'Assamese': 'শেষ বাৰ কথা পতাৰ পিছত আপোনাৰ লক্ষণ কেনেকৈ সলনি হ\'ল?',
        'Urdu': 'آخری بار بات کرنے کے بعد آپ کی علامات میں کیا تبدیلی آئی؟'
    },
    'day3_q2': {
        'English': 'Have you consulted a doctor about these symptoms?',
        'Hindi': 'क्या आपने इन लक्षणों के बारे में डॉक्टर से बात की?',
        'Telugu': 'ఈ లక్షణాల గురించి మీరు డాక్టర్‌ను సంప్రదించారా?',
        'Bengali': 'এই লক্ষণ নিয়ে ডাক্তারের সাথে কথা বলেছেন?',
        'Marathi': 'या लक्षणांबद्दल तुम्ही डॉक्टरांशी बोललात का?',
        'Tamil': 'இந்த அறிகுறிகள் பற்றி மருத்துவரிடம் ஆலோசித்தீர்களா?',
        'Gujarati': 'આ લક્ષણો વિશે ડોક્ટર સાથે વાત કરી?',
        'Kannada': 'ಈ ಲಕ್ಷಣಗಳ ಬಗ್ಗೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿದ್ದೀರಾ?',
        'Malayalam': 'ഈ ലക്ഷണങ്ങളെക്കുറിച്ച് ഡോക്ടറെ കണ്ടോ?',
        'Punjabi': 'ਕੀ ਤੁਸੀਂ ਇਹਨਾਂ ਲੱਛਣਾਂ ਬਾਰੇ ਡਾਕਟਰ ਨਾਲ ਗੱਲ ਕੀਤੀ?',
        'Odia': 'ଏହି ଲକ୍ଷଣ ବିଷୟରେ ଆପଣ ଡାକ୍ତରଙ୍କ ସହ କଥା ହୋଇଛନ୍ତି କି?',
        'Assamese': 'এই লক্ষণবোৰৰ বিষয়ে ডাক্তৰৰ লগত কথা পাতিছে নে?',
        'Urdu': 'کیا آپ نے ان علامات کے بارے میں ڈاکٹر سے بات کی؟'
    },
    'day3_q3': {
        'English': 'Are you still taking the medication?',
        'Hindi': 'क्या आप अभी भी दवाई ले रहे हैं?',
        'Telugu': 'మీరు ఇంకా మందు తీసుకుంటున్నారా?',
        'Bengali': 'আপনি কি এখনও ওষুধ খাচ্ছেন?',
        'Marathi': 'तुम्ही अजूनही औषध घेत आहात का?',
        'Tamil': 'நீங்கள் இன்னும் மருந்து எடுக்கிறீர்களா?',
        'Gujarati': 'તમે હજુ પણ દવા લઈ રહ્યા છો?',
        'Kannada': 'ನೀವು ಇನ್ನೂ ಔಷಧಿ ತೆಗೆದುಕೊಳ್ಳುತ್ತಿದ್ದೀರಾ?',
        'Malayalam': 'നിങ്ങൾ ഇപ്പോഴും മരുന്ന് കഴിക്കുന്നുണ്ടോ?',
        'Punjabi': 'ਕੀ ਤੁਸੀਂ ਅਜੇ ਵੀ ਦਵਾਈ ਲੈ ਰਹੇ ਹੋ?',
        'Odia': 'ଆପଣ ଏବେ ବି ଔଷଧ ଖାଉଛନ୍ତି କି?',
        'Assamese': 'আপুনি এতিয়াও ঔষধ খাই আছে নে?',
        'Urdu': 'کیا آپ ابھی بھی دوا لے رہے ہیں؟'
    },
    'day3_q4': {
        'English': 'Have you noticed any new symptoms?',
        'Hindi': 'क्या आपने कोई नया लक्षण देखा?',
        'Telugu': 'మీరు ఏదైనా కొత్త లక్షణాలు గమనించారా?',
        'Bengali': 'নতুন কোনো লক্ষণ দেখেছেন কি?',
        'Marathi': 'कोणती नवीन लक्षणे दिसली का?',
        'Tamil': 'புதிய அறிகுறிகள் கவனித்தீர்களா?',
        'Gujarati': 'કોઈ નવા લક્ષણ જોયા?',
        'Kannada': 'ಯಾವುದಾದರೂ ಹೊಸ ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿದಿರಾ?',
        'Malayalam': 'പുതിയ ലക്ഷണങ്ങൾ ശ്രദ്ധിച്ചോ?',
        'Punjabi': 'ਕੋਈ ਨਵੇਂ ਲੱਛਣ ਦੇਖੇ?',
        'Odia': 'କୌଣସି ନୂଆ ଲକ୍ଷଣ ଦେଖିଛନ୍ତି କି?',
        'Assamese': 'কোনো নতুন লক্ষণ দেখিছে নে?',
        'Urdu': 'کوئی نئی علامت دیکھی؟'
    },
    # Day 5 Questions
    'day5_q1': {
        'English': 'Are your symptoms improving, staying the same, or getting worse?',
        'Hindi': 'क्या आपके लक्षण बेहतर हो रहे हैं, वैसे ही हैं, या बिगड़ रहे हैं?',
        'Telugu': 'మీ లక్షణాలు మెరుగుపడుతున్నాయా, అలాగే ఉన్నాయా, లేదా మరింత తీవ్రమవుతున్నాయా?',
        'Bengali': 'আপনার লক্ষণ ভালো হচ্ছে, একই আছে, নাকি খারাপ হচ্ছে?',
        'Marathi': 'तुमची लक्षणे सुधारत आहेत, तशीच आहेत, की वाढत आहेत?',
        'Tamil': 'உங்கள் அறிகுறிகள் மேம்படுகிறதா, அப்படியே இருக்கிறதா, அல்லது மோசமாகிறதா?',
        'Gujarati': 'તમારા લક્ષણો સુધરી રહ્યા છે, એવા જ છે, કે વધુ ખરાબ થઈ રહ્યા છે?',
        'Kannada': 'ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಸುಧಾರಿಸುತ್ತಿವೆಯೇ, ಹಾಗೆಯೇ ಇವೆಯೇ, ಅಥವಾ ಹದಗೆಡುತ್ತಿವೆಯೇ?',
        'Malayalam': 'നിങ്ങളുടെ ലക്ഷണങ്ങൾ മെച്ചപ്പെടുന്നുണ്ടോ, അതേപടി ആണോ, അതോ വഷളാവുന്നുണ്ടോ?',
        'Punjabi': 'ਤੁਹਾਡੇ ਲੱਛਣ ਸੁਧਰ ਰਹੇ ਹਨ, ਉਸੇ ਤਰ੍ਹਾਂ ਹਨ, ਜਾਂ ਵਿਗੜ ਰਹੇ ਹਨ?',
        'Odia': 'ଆପଣଙ୍କ ଲକ୍ଷଣ ଉନ୍ନତ ହେଉଛି, ଏପରି ଅଛି, ନା ଖରାପ ହେଉଛି?',
        'Assamese': 'আপোনাৰ লক্ষণ ভাল হৈ আছে, একেই আছে, নে বেয়া হৈ আছে?',
        'Urdu': 'کیا آپ کی علامات بہتر ہو رہی ہیں، ویسی ہی ہیں، یا خراب ہو رہی ہیں؟'
    },
    'day5_q2': {
        'English': 'Did you need to visit a hospital due to these symptoms?',
        'Hindi': 'क्या इन लक्षणों की वजह से आपको अस्पताल जाना पड़ा?',
        'Telugu': 'ఈ లక్షణాల వల్ల మీరు ఆసుపత్రికి వెళ్ళాల్సి వచ్చిందా?',
        'Bengali': 'এই লক্ষণের জন্য হাসপাতালে যেতে হয়েছিল?',
        'Marathi': 'या लक्षणांमुळे तुम्हाला हॉस्पिटलला जावे लागले का?',
        'Tamil': 'இந்த அறிகுறிகளால் மருத்துவமனைக்கு செல்ல வேண்டியிருந்ததா?',
        'Gujarati': 'આ લક્ષણોને કારણે હોસ્પિટલ જવું પડ્યું?',
        'Kannada': 'ಈ ಲಕ್ಷಣಗಳಿಂದಾಗಿ ಆಸ್ಪತ್ರೆಗೆ ಹೋಗಬೇಕಾಯಿತೇ?',
        'Malayalam': 'ഈ ലക്ഷണങ്ങൾ കാരണം ആശുപത്രിയിൽ പോകേണ്ടി വന്നോ?',
        'Punjabi': 'ਇਹਨਾਂ ਲੱਛਣਾਂ ਕਰਕੇ ਹਸਪਤਾਲ ਜਾਣਾ ਪਿਆ?',
        'Odia': 'ଏହି ଲକ୍ଷଣ ପାଇଁ ହସ୍ପିଟାଲ ଯିବାକୁ ପଡ଼ିଲା କି?',
        'Assamese': 'এই লক্ষণৰ বাবে হাস্পতাললৈ যাব লগা হ\'ল নে?',
        'Urdu': 'کیا ان علامات کی وجہ سے ہسپتال جانا پڑا؟'
    },
    'day5_q3': {
        'English': 'How have these symptoms affected your daily activities?',
        'Hindi': 'इन लक्षणों ने आपकी दैनिक गतिविधियों को कैसे प्रभावित किया?',
        'Telugu': 'ఈ లక్షణాలు మీ రోజువారీ కార్యకలాపాలను ఎలా ప్రభావితం చేశాయి?',
        'Bengali': 'এই লক্ষণ আপনার দৈনন্দিন কাজকর্মে কতটা প্রভাব ফেলেছে?',
        'Marathi': 'या लक्षणांमुळे तुमच्या दैनंदिन कामांवर कसा परिणाम झाला?',
        'Tamil': 'இந்த அறிகுறிகள் உங்கள் அன்றாட செயல்பாடுகளை எவ்வாறு பாதித்தன?',
        'Gujarati': 'આ લક્ષણોએ તમારી રોજિંદી પ્રવૃત્તિઓને કેવી અસર કરી?',
        'Kannada': 'ಈ ಲಕ್ಷಣಗಳು ನಿಮ್ಮ ದೈನಂದಿನ ಚಟುವಟಿಕೆಗಳ ಮೇಲೆ ಹೇಗೆ ಪರಿಣಾಮ ಬೀರಿದವು?',
        'Malayalam': 'ഈ ലക്ഷണങ്ങൾ നിങ്ങളുടെ ദൈനംദിന പ്രവർത്തനങ്ങളെ എങ്ങനെ ബാധിച്ചു?',
        'Punjabi': 'ਇਹ ਲੱਛਣਾਂ ਨੇ ਤੁਹਾਡੀਆਂ ਰੋਜ਼ਾਨਾ ਗਤੀਵਿਧੀਆਂ ਨੂੰ ਕਿਵੇਂ ਪ੍ਰਭਾਵਿਤ ਕੀਤਾ?',
        'Odia': 'ଏହି ଲକ୍ଷଣଗୁଡ଼ିକ ଆପଣଙ୍କ ଦୈନିକ କାର୍ଯ୍ୟକୁ କେମିତି ପ୍ରଭାବିତ କରିଛି?',
        'Assamese': 'এই লক্ষণে আপোনাৰ দৈনিক কামত কেনে প্ৰভাৱ পেলাইছে?',
        'Urdu': 'ان علامات نے آپ کی روزمرہ سرگرمیوں کو کیسے متاثر کیا؟'
    },
    'day5_q4': {
        'English': 'Have you taken any other medications to manage these symptoms?',
        'Hindi': 'क्या आपने इन लक्षणों के लिए कोई अन्य दवाई ली?',
        'Telugu': 'ఈ లక్షణాలను నిర్వహించడానికి మీరు ఇతర మందులు తీసుకున్నారా?',
        'Bengali': 'এই লক্ষণ সামলাতে অন্য কোনো ওষুধ খেয়েছেন?',
        'Marathi': 'या लक्षणांसाठी इतर कोणतीही औषधे घेतली का?',
        'Tamil': 'இந்த அறிகுறிகளுக்கு வேறு மருந்துகள் எடுத்தீர்களா?',
        'Gujarati': 'આ લક્ષણો માટે બીજી કોઈ દવા લીધી?',
        'Kannada': 'ಈ ಲಕ್ಷಣಗಳಿಗಾಗಿ ಬೇರೆ ಯಾವುದಾದರೂ ಔಷಧಿ ತೆಗೆದುಕೊಂಡಿರಾ?',
        'Malayalam': 'ഈ ലക്ഷണങ്ങൾക്ക് മറ്റ് മരുന്നുകൾ കഴിച്ചോ?',
        'Punjabi': 'ਇਹਨਾਂ ਲੱਛਣਾਂ ਲਈ ਕੋਈ ਹੋਰ ਦਵਾਈ ਲਈ?',
        'Odia': 'ଏହି ଲକ୍ଷଣ ପାଇଁ ଅନ୍ୟ କୌଣସି ଔଷଧ ନେଇଛନ୍ତି କି?',
        'Assamese': 'এই লক্ষণৰ বাবে আন কোনো ঔষধ খাইছে নে?',
        'Urdu': 'ان علامات کے لیے کوئی اور دوا لی؟'
    },
    # Day 7 Questions
    'day7_q1': {
        'English': 'What is the current status of your symptoms?',
        'Hindi': 'आपके लक्षणों की वर्तमान स्थिति क्या है?',
        'Telugu': 'మీ లక్షణాల ప్రస్తుత స్థితి ఏమిటి?',
        'Bengali': 'আপনার লক্ষণের বর্তমান অবস্থা কী?',
        'Marathi': 'तुमच्या लक्षणांची सध्याची स्थिती काय आहे?',
        'Tamil': 'உங்கள் அறிகுறிகளின் தற்போதைய நிலை என்ன?',
        'Gujarati': 'તમારા લક્ષણોની હાલની સ્થિતિ શું છે?',
        'Kannada': 'ನಿಮ್ಮ ಲಕ್ಷಣಗಳ ಪ್ರಸ್ತುತ ಸ್ಥಿತಿ ಏನು?',
        'Malayalam': 'നിങ്ങളുടെ ലക്ഷണങ്ങളുടെ ഇപ്പോഴത്തെ അവസ്ഥ എന്താണ്?',
        'Punjabi': 'ਤੁਹਾਡੇ ਲੱਛਣਾਂ ਦੀ ਮੌਜੂਦਾ ਸਥਿਤੀ ਕੀ ਹੈ?',
        'Odia': 'ଆପଣଙ୍କ ଲକ୍ଷଣର ବର୍ତ୍ତମାନ ଅବସ୍ଥା କଣ?',
        'Assamese': 'আপোনাৰ লক্ষণৰ বৰ্তমান অৱস্থা কি?',
        'Urdu': 'آپ کی علامات کی موجودہ حالت کیا ہے؟'
    },
    'day7_q2': {
        'English': 'If your symptoms have resolved, when did they stop?',
        'Hindi': 'अगर आपके लक्षण ठीक हो गए हैं, तो कब रुके?',
        'Telugu': 'మీ లక్షణాలు పరిష్కారమైతే, అవి ఎప్పుడు ఆగాయి?',
        'Bengali': 'লক্ষণ সেরে গেলে, কবে থামল?',
        'Marathi': 'लक्षणे बरी झाली असल्यास, ती कधी थांबली?',
        'Tamil': 'அறிகுறிகள் தீர்ந்தால், அவை எப்போது நின்றன?',
        'Gujarati': 'લક્ષણો ઠીક થઈ ગયા હોય તો, ક્યારે બંધ થયા?',
        'Kannada': 'ಲಕ್ಷಣಗಳು ಸರಿಯಾಗಿದ್ದರೆ, ಯಾವಾಗ ನಿಂತವು?',
        'Malayalam': 'ലക്ഷണങ്ങൾ ഭേദമായെങ്കിൽ, അവ എപ്പോൾ നിന്നു?',
        'Punjabi': 'ਜੇ ਲੱਛਣ ਠੀਕ ਹੋ ਗਏ, ਕਦੋਂ ਬੰਦ ਹੋਏ?',
        'Odia': 'ଲକ୍ଷଣ ଭଲ ହୋଇଗଲେ, ସେଗୁଡ଼ିକ କେବେ ବନ୍ଦ ହେଲା?',
        'Assamese': 'লক্ষণ ভাল হ\'লে, কেতিয়া বন্ধ হ\'ল?',
        'Urdu': 'اگر علامات ٹھیک ہو گئیں، تو کب رکیں؟'
    },
    'day7_q3': {
        'English': 'Would you like us to arrange a free health check-up?',
        'Hindi': 'क्या आप चाहते हैं कि हम मुफ्त स्वास्थ्य जांच की व्यवस्था करें?',
        'Telugu': 'మేము ఉచిత ఆరోగ్య పరీక్ష ఏర్పాటు చేయమంటారా?',
        'Bengali': 'আমরা কি বিনামূল্যে স্বাস্থ্য পরীক্ষার ব্যবস্থা করব?',
        'Marathi': 'आम्ही मोफत आरोग्य तपासणी आयोजित करावी का?',
        'Tamil': 'இலவச உடல்நலப் பரிசோதனை ஏற்பாடு செய்யவா?',
        'Gujarati': 'અમે મફત આરોગ્ય તપાસની વ્યવસ્થા કરીએ?',
        'Kannada': 'ನಾವು ಉಚಿತ ಆರೋಗ್ಯ ತಪಾಸಣೆ ಏರ್ಪಡಿಸಬೇಕೇ?',
        'Malayalam': 'സൗജന്യ ആരോഗ്യ പരിശോധന ഏർപ്പാട് ചെയ്യണോ?',
        'Punjabi': 'ਕੀ ਅਸੀਂ ਮੁਫ਼ਤ ਸਿਹਤ ਜਾਂਚ ਦਾ ਪ੍ਰਬੰਧ ਕਰੀਏ?',
        'Odia': 'ଆମେ ମାଗଣା ସ୍ୱାସ୍ଥ୍ୟ ପରୀକ୍ଷା କରିବୁ କି?',
        'Assamese': 'আমি বিনামূলীয়া স্বাস্থ্য পৰীক্ষাৰ ব্যৱস্থা কৰোঁ নে?',
        'Urdu': 'کیا ہم مفت صحت کی جانچ کا بندوبست کریں؟'
    },
    'day7_q4': {
        'English': 'Is there anything else you would like to share about your experience?',
        'Hindi': 'क्या आप अपने अनुभव के बारे में कुछ और साझा करना चाहेंगे?',
        'Telugu': 'మీ అనుభవం గురించి ఇంకేదైనా చెప్పాలనుకుంటున్నారా?',
        'Bengali': 'আপনার অভিজ্ঞতা সম্পর্কে আর কিছু বলতে চান?',
        'Marathi': 'तुमच्या अनुभवाबद्दल आणखी काही सांगायचे आहे का?',
        'Tamil': 'உங்கள் அனுபவத்தைப் பற்றி வேறு ஏதாவது பகிர விரும்புகிறீர்களா?',
        'Gujarati': 'તમારા અનુભવ વિશે બીજું કંઈ કહેવું છે?',
        'Kannada': 'ನಿಮ್ಮ ಅನುಭವದ ಬಗ್ಗೆ ಬೇರೆ ಏನಾದರೂ ಹಂಚಿಕೊಳ್ಳಲು ಇಷ್ಟಪಡುತ್ತೀರಾ?',
        'Malayalam': 'നിങ്ങളുടെ അനുഭവത്തെക്കുറിച്ച് മറ്റെന്തെങ്കിലും പറയാനുണ്ടോ?',
        'Punjabi': 'ਆਪਣੇ ਅਨੁਭਵ ਬਾਰੇ ਹੋਰ ਕੁਝ ਦੱਸਣਾ ਚਾਹੁੰਦੇ ਹੋ?',
        'Odia': 'ଆପଣଙ୍କ ଅନୁଭୂତି ବିଷୟରେ ଆଉ କିଛି କହିବାକୁ ଚାହୁଁଛନ୍ତି କି?',
        'Assamese': 'আপোনাৰ অভিজ্ঞতাৰ বিষয়ে আৰু কিবা ক\'বলৈ আছে নে?',
        'Urdu': 'کیا آپ اپنے تجربے کے بارے میں کچھ اور بتانا چاہیں گے؟'
    },
    # Fallback question translations (when LLM is unavailable)
    'fb_day1_q1': {
        'English': 'Can you describe your symptoms in more detail?',
        'Hindi': 'क्या आप अपने लक्षणों का अधिक विस्तार से वर्णन कर सकते हैं?',
        'Telugu': 'మీ లక్షణాలను మరింత వివరంగా వివరించగలరా?',
        'Bengali': 'আপনি কি আপনার লক্ষণগুলি আরও বিস্তারিতভাবে বর্ণনা করতে পারেন?',
        'Marathi': 'तुम्ही तुमच्या लक्षणांचे अधिक तपशीलवार वर्णन करू शकता का?',
        'Tamil': 'உங்கள் அறிகுறிகளை மேலும் விரிவாக விவரிக்க முடியுமா?',
        'Gujarati': 'શું તમે તમારા લક્ષણોનું વધુ વિગતવાર વર્ણન કરી શકો છો?',
        'Kannada': 'ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಹೆಚ್ಚು ವಿವರವಾಗಿ ವಿವರಿಸಬಹುದೇ?',
        'Malayalam': 'നിങ്ങളുടെ ലക്ഷണങ്ങൾ കൂടുതൽ വിശദമായി വിവരിക്കാമോ?',
        'Punjabi': 'ਕੀ ਤੁਸੀਂ ਆਪਣੇ ਲੱਛਣਾਂ ਦਾ ਹੋਰ ਵਿਸਤਾਰ ਨਾਲ ਵਰਣਨ ਕਰ ਸਕਦੇ ਹੋ?',
        'Odia': 'ଆପଣ ଆପଣଙ୍କ ଲକ୍ଷଣଗୁଡ଼ିକୁ ଅଧିକ ବିସ୍ତୃତ ଭାବରେ ବର୍ଣ୍ଣନା କରିପାରିବେ କି?',
        'Assamese': 'আপুনি আপোনাৰ লক্ষণবোৰ অধিক বিস্তৃতভাৱে বৰ্ণনা কৰিব পাৰিবনে?',
        'Urdu': 'کیا آپ اپنی علامات کی مزید تفصیل سے وضاحت کر سکتے ہیں؟'
    },
    'fb_day1_q2': {
        'English': 'Are you experiencing any discomfort right now?',
        'Hindi': 'क्या आप अभी कोई तकलीफ महसूस कर रहे हैं?',
        'Telugu': 'మీకు ఇప్పుడు ఏదైనా అసౌకర్యం అనుభవమవుతోందా?',
        'Bengali': 'আপনি কি এখন কোনো অস্বস্তি অনুভব করছেন?',
        'Marathi': 'तुम्हाला आत्ता कोणत्याही अस्वस्थता जाणवत आहे का?',
        'Tamil': 'இப்போது ஏதேனும் அசௌகரியம் உணர்கிறீர்களா?',
        'Gujarati': 'શું તમે હાલમાં કોઈ અસ્વસ્થતા અનુભવી રહ્યા છો?',
        'Kannada': 'ನೀವು ಈಗ ಯಾವುದಾದರೂ ಅಸ್ವಸ್ಥತೆ ಅನುಭವಿಸುತ್ತಿದ್ದೀರಾ?',
        'Malayalam': 'നിങ്ങൾക്ക് ഇപ്പോൾ എന്തെങ്കിലും അസ്വസ്ഥത അനുഭവപ്പെടുന്നുണ്ടോ?',
        'Punjabi': 'ਕੀ ਤੁਸੀਂ ਇਸ ਸਮੇਂ ਕੋਈ ਤਕਲੀਫ ਮਹਿਸੂਸ ਕਰ ਰਹੇ ਹੋ?',
        'Odia': 'ଆପଣ ଏବେ କୌଣସି ଅସୁବିଧା ଅନୁଭବ କରୁଛନ୍ତି କି?',
        'Assamese': 'আপুনি এতিয়া কোনো অস্বস্তি অনুভৱ কৰি আছে নেকি?',
        'Urdu': 'کیا آپ ابھی کوئی تکلیف محسوس کر رہے ہیں؟'
    },
    'fb_day1_onset': {
        'English': 'When did you first notice these symptoms?',
        'Hindi': 'आपने पहली बार ये लक्षण कब नोटिस किए?',
        'Telugu': 'మీరు ఈ లక్షణాలను మొదట ఎప్పుడు గమనించారు?',
        'Bengali': 'আপনি প্রথম কবে এই লক্ষণগুলি লক্ষ্য করেছিলেন?',
        'Marathi': 'तुम्हाला ही लक्षणे पहिल्यांदा कधी दिसली?',
        'Tamil': 'இந்த அறிகுறிகளை நீங்கள் முதலில் எப்போது கவனித்தீர்கள்?',
        'Gujarati': 'તમે આ લક્ષણો પહેલીવાર ક્યારે જોયા?',
        'Kannada': 'ನೀವು ಈ ಲಕ್ಷಣಗಳನ್ನು ಮೊದಲು ಯಾವಾಗ ಗಮನಿಸಿದಿರಿ?',
        'Malayalam': 'ഈ ലക്ഷണങ്ങൾ നിങ്ങൾ ആദ്യം എപ്പോൾ ശ്രദ്ധിച്ചു?',
        'Punjabi': 'ਤੁਸੀਂ ਇਹ ਲੱਛਣ ਪਹਿਲੀ ਵਾਰ ਕਦੋਂ ਦੇਖੇ?',
        'Odia': 'ଏହି ଲକ୍ଷଣଗୁଡ଼ିକ ଆପଣ ପ୍ରଥମେ କେବେ ଲକ୍ଷ୍ୟ କରିଥିଲେ?',
        'Assamese': 'এই লক্ষণবোৰ আপুনি প্ৰথমে কেতিয়া দেখিছিল?',
        'Urdu': 'آپ نے یہ علامات پہلی بار کب محسوس کیں؟'
    },
    'fb_day3_q1': {
        'English': 'Have your symptoms improved since you first reported them?',
        'Hindi': 'क्या पहली बार रिपोर्ट करने के बाद से आपके लक्षणों में सुधार हुआ है?',
        'Telugu': 'మీరు మొదట రిపోర్ట్ చేసినప్పటి నుండి మీ లక్షణాలు మెరుగుపడ్డాయా?',
        'Bengali': 'প্রথম রিপোর্ট করার পর থেকে আপনার লক্ষণগুলি উন্নতি হয়েছে?',
        'Marathi': 'पहिल्यांदा सांगितल्यापासून तुमच्या लक्षणांमध्ये सुधारणा झाली का?',
        'Tamil': 'முதலில் தெரிவித்ததிலிருந்து உங்கள் அறிகுறிகள் மேம்பட்டதா?',
        'Gujarati': 'પ્રથમ વખત જણાવ્યા પછી તમારા લક્ષણોમાં સુધારો થયો છે?',
        'Kannada': 'ಮೊದಲು ವರದಿ ಮಾಡಿದ ನಂತರ ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಸುಧಾರಿಸಿವೆಯೇ?',
        'Malayalam': 'ആദ്യം റിപ്പോർട്ട് ചെയ്തതിന് ശേഷം നിങ്ങളുടെ ലക്ഷണങ്ങൾ മെച്ചപ്പെട്ടോ?',
        'Punjabi': 'ਪਹਿਲੀ ਵਾਰ ਦੱਸਣ ਤੋਂ ਬਾਅਦ ਤੁਹਾਡੇ ਲੱਛਣਾਂ ਵਿੱਚ ਸੁਧਾਰ ਹੋਇਆ ਹੈ?',
        'Odia': 'ପ୍ରଥମ ଥର ଜଣାଇବା ପରେ ଆପଣଙ୍କ ଲକ୍ଷଣରେ ସୁଧାର ହୋଇଛି କି?',
        'Assamese': 'প্ৰথম বাৰ কোৱাৰ পিছত আপোনাৰ লক্ষণ উন্নতি হৈছে নে?',
        'Urdu': 'پہلی بار بتانے کے بعد سے آپ کی علامات میں بہتری آئی ہے؟'
    },
    'fb_day3_q2': {
        'English': 'Did you need to take any action to manage your symptoms?',
        'Hindi': 'क्या आपको अपने लक्षणों को संभालने के लिए कोई कदम उठाना पड़ा?',
        'Telugu': 'మీ లక్షణాలను నిర్వహించడానికి మీరు ఏదైనా చర్య తీసుకోవలసి వచ్చిందా?',
        'Bengali': 'আপনার লক্ষণ সামলাতে কোনো পদক্ষেপ নিতে হয়েছে?',
        'Marathi': 'तुमच्या लक्षणांचे व्यवस्थापन करण्यासाठी तुम्हाला काही कृती करावी लागली का?',
        'Tamil': 'உங்கள் அறிகுறிகளை சமாளிக்க ஏதேனும் நடவடிக்கை எடுக்க வேண்டியிருந்ததா?',
        'Gujarati': 'તમારા લક્ષણોનું સંચાલન કરવા માટે કોઈ પગલું લેવું પડ્યું?',
        'Kannada': 'ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ನಿರ್ವಹಿಸಲು ಯಾವುದಾದರೂ ಕ್ರಮ ತೆಗೆದುಕೊಳ್ಳಬೇಕಾಯಿತೇ?',
        'Malayalam': 'നിങ്ങളുടെ ലക്ഷണങ്ങൾ കൈകാര്യം ചെയ്യാൻ എന്തെങ്കിലും നടപടി എടുക്കേണ്ടി വന്നോ?',
        'Punjabi': 'ਕੀ ਤੁਹਾਨੂੰ ਆਪਣੇ ਲੱਛਣਾਂ ਨੂੰ ਸੰਭਾਲਣ ਲਈ ਕੋਈ ਕਦਮ ਚੁੱਕਣਾ ਪਿਆ?',
        'Odia': 'ଆପଣଙ୍କ ଲକ୍ଷଣ ସମ୍ଭାଳିବାକୁ କୌଣସି ପଦକ୍ଷେପ ନେବାକୁ ପଡ଼ିଲା କି?',
        'Assamese': 'আপোনাৰ লক্ষণ সামৰিবলৈ কোনো পদক্ষেপ ল\'বলগীয়া হ\'ল নে?',
        'Urdu': 'کیا آپ کو اپنی علامات کو سنبھالنے کے لیے کوئی قدم اٹھانا پڑا؟'
    },
    'fb_day5_q1': {
        'English': 'How are you managing with these symptoms on a daily basis?',
        'Hindi': 'आप रोजाना इन लक्षणों को कैसे संभाल रहे हैं?',
        'Telugu': 'మీరు ప్రతిరోజు ఈ లక్షణాలతో ఎలా నిర్వహిస్తున్నారు?',
        'Bengali': 'প্রতিদিন এই লক্ষণগুলি কীভাবে সামলাচ্ছেন?',
        'Marathi': 'तुम्ही दररोज या लक्षणांना कसे हाताळत आहात?',
        'Tamil': 'இந்த அறிகுறிகளை தினமும் எவ்வாறு சமாளிக்கிறீர்கள்?',
        'Gujarati': 'તમે દરરોજ આ લક્ષણોને કેવી રીતે સંભાળી રહ્યા છો?',
        'Kannada': 'ಈ ಲಕ್ಷಣಗಳನ್ನು ಪ್ರತಿದಿನ ಹೇಗೆ ನಿರ್ವಹಿಸುತ್ತಿದ್ದೀರಿ?',
        'Malayalam': 'ഈ ലക്ഷണങ്ങളെ ദിവസേന എങ്ങനെ കൈകാര്യം ചെയ്യുന്നു?',
        'Punjabi': 'ਤੁਸੀਂ ਰੋਜ਼ਾਨਾ ਇਹ ਲੱਛਣਾਂ ਨੂੰ ਕਿਵੇਂ ਸੰਭਾਲ ਰਹੇ ਹੋ?',
        'Odia': 'ଆପଣ ପ୍ରତିଦିନ ଏହି ଲକ୍ଷଣଗୁଡ଼ିକୁ କେମିତି ସମ୍ଭାଳୁଛନ୍ତି?',
        'Assamese': 'এই লক্ষণবোৰ প্ৰতিদিনে কেনেকৈ সামৰি আছে?',
        'Urdu': 'آپ روزانہ ان علامات کو کیسے سنبھال رہے ہیں؟'
    },
    'fb_day5_q2': {
        'English': 'Have you noticed any patterns in when symptoms occur?',
        'Hindi': 'क्या आपने देखा कि लक्षण कब होते हैं इसमें कोई पैटर्न है?',
        'Telugu': 'లక్షణాలు ఎప్పుడు వస్తాయనే దానిలో ఏదైనా ప్యాటర్న్ గమనించారా?',
        'Bengali': 'লক্ষণ কখন হয় তার কোনো প্যাটার্ন দেখেছেন?',
        'Marathi': 'लक्षणे कधी येतात यात काही पॅटर्न दिसला का?',
        'Tamil': 'அறிகுறிகள் எப்போது வருகின்றன என்பதில் ஏதேனும் முறை கவனித்தீர்களா?',
        'Gujarati': 'લક્ષણો ક્યારે આવે છે તેમાં કોઈ પેટર્ન જોયું છે?',
        'Kannada': 'ಲಕ್ಷಣಗಳು ಯಾವಾಗ ಬರುತ್ತವೆ ಎಂಬುದರಲ್ಲಿ ಯಾವುದೇ ಮಾದರಿ ಗಮನಿಸಿದ್ದೀರಾ?',
        'Malayalam': 'ലക്ഷണങ്ങൾ എപ്പോൾ വരുന്നു എന്നതിൽ എന്തെങ്കിലും പാറ്റേൺ ശ്രദ്ധിച്ചോ?',
        'Punjabi': 'ਕੀ ਤੁਸੀਂ ਦੇਖਿਆ ਕਿ ਲੱਛਣ ਕਦੋਂ ਆਉਂਦੇ ਹਨ ਇਸ ਵਿੱਚ ਕੋਈ ਪੈਟਰਨ ਹੈ?',
        'Odia': 'ଲକ୍ଷଣ କେବେ ହୁଏ ଏଥିରେ କୌଣସି ପ୍ୟାଟର୍ନ ଲକ୍ଷ୍ୟ କରିଛନ୍ତି କି?',
        'Assamese': 'লক্ষণ কেতিয়া হয় তাৰ কোনো পেটাৰ্ন দেখিছে নে?',
        'Urdu': 'کیا آپ نے دیکھا کہ علامات کب آتی ہیں اس میں کوئی پیٹرن ہے؟'
    },
    'fb_day7_q1': {
        'English': 'Looking back over the past week, how has your condition changed?',
        'Hindi': 'पिछले हफ्ते को देखते हुए, आपकी स्थिति में क्या बदलाव आया है?',
        'Telugu': 'గత వారాన్ని తిరిగి చూస్తే, మీ పరిస్థితి ఎలా మారింది?',
        'Bengali': 'গত সপ্তাহ দেখলে, আপনার অবস্থা কীভাবে বদলেছে?',
        'Marathi': 'मागील आठवडा पाहता, तुमची स्थिती कशी बदलली?',
        'Tamil': 'கடந்த வாரத்தை திரும்பிப் பார்க்கையில், உங்கள் நிலை எவ்வாறு மாறியது?',
        'Gujarati': 'છેલ્લા અઠવાડિયાને જોતાં, તમારી સ્થિતિ કેવી રીતે બદલાઈ?',
        'Kannada': 'ಕಳೆದ ವಾರವನ್ನು ಹಿಂತಿರುಗಿ ನೋಡಿದರೆ, ನಿಮ್ಮ ಸ್ಥಿತಿ ಹೇಗೆ ಬದಲಾಯಿತು?',
        'Malayalam': 'കഴിഞ്ഞ ആഴ്ച തിരിഞ്ഞുനോക്കിയാൽ, നിങ്ങളുടെ അവസ്ഥ എങ്ങനെ മാറി?',
        'Punjabi': 'ਪਿਛਲੇ ਹਫ਼ਤੇ ਨੂੰ ਦੇਖਦਿਆਂ, ਤੁਹਾਡੀ ਹਾਲਤ ਕਿਵੇਂ ਬਦਲੀ?',
        'Odia': 'ଗତ ସପ୍ତାହକୁ ଦେଖିଲେ, ଆପଣଙ୍କ ଅବସ୍ଥା କେମିତି ବଦଳିଛି?',
        'Assamese': 'যোৱা সপ্তাহ চালে, আপোনাৰ অৱস্থা কেনেকৈ সলনি হ\'ল?',
        'Urdu': 'پچھلے ہفتے کو دیکھتے ہوئے، آپ کی حالت کیسے بدلی؟'
    },
    'fb_day7_q2': {
        'English': 'Is there anything you wish we had asked about earlier?',
        'Hindi': 'क्या कुछ ऐसा है जो आप चाहते हैं कि हमने पहले पूछा होता?',
        'Telugu': 'మేము ముందుగా అడగాలని మీరు కోరుకునేది ఏదైనా ఉందా?',
        'Bengali': 'এমন কিছু আছে যা আমরা আগে জিজ্ঞেস করলে ভালো হতো?',
        'Marathi': 'आम्ही आधी विचारले असते असे काही आहे का?',
        'Tamil': 'நாங்கள் முன்பே கேட்டிருக்க வேண்டும் என்று நீங்கள் நினைக்கும் ஏதாவது இருக்கிறதா?',
        'Gujarati': 'અમે પહેલા પૂછ્યું હોત તો સારું થાત એવું કંઈ છે?',
        'Kannada': 'ನಾವು ಮೊದಲೇ ಕೇಳಬೇಕಾಗಿತ್ತು ಎಂದು ನೀವು ಬಯಸುವ ಏನಾದರೂ ಇದೆಯೇ?',
        'Malayalam': 'ഞങ്ങൾ നേരത്തെ ചോദിച്ചിരുന്നെങ്കിൽ എന്ന് നിങ്ങൾ ആഗ്രഹിക്കുന്ന എന്തെങ്കിലും ഉണ്ടോ?',
        'Punjabi': 'ਕੀ ਕੁਝ ਅਜਿਹਾ ਹੈ ਜੋ ਅਸੀਂ ਪਹਿਲਾਂ ਪੁੱਛਿਆ ਹੁੰਦਾ ਤਾਂ ਚੰਗਾ ਹੁੰਦਾ?',
        'Odia': 'ଆମେ ଆଗରୁ ପଚାରିଥାନ୍ତେ ବୋଲି ଆପଣ ଚାହୁଁଥିବା କିଛି ଅଛି କି?',
        'Assamese': 'আমি আগতে সুধিলে ভাল হ\'লহেঁতেন বুলি আপুনি ভবা কিবা আছে নে?',
        'Urdu': 'کیا کچھ ایسا ہے جو آپ چاہتے ہیں کہ ہم نے پہلے پوچھا ہوتا؟'
    }
}

# Multi-language option translations
OPTION_TRANSLATIONS = {
    'feeling_fine': {
        'English': 'Feeling fine, no issues',
        'Hindi': 'ठीक महसूस कर रहा/रही हूँ, कोई समस्या नहीं',
        'Telugu': 'బాగానే ఉన్నాను, ఏ సమస్య లేదు',
        'Bengali': 'ভালো লাগছে, কোনো সমস্যা নেই',
        'Marathi': 'ठीक वाटतंय, कोणतीही समस्या नाही',
        'Tamil': 'நன்றாக உணர்கிறேன், எந்த பிரச்சனையும் இல்லை',
        'Gujarati': 'સારું લાગે છે, કોઈ સમસ્યા નથી',
        'Kannada': 'ಚೆನ್ನಾಗಿದ್ದೇನೆ, ಯಾವುದೇ ಸಮಸ್ಯೆ ಇಲ್ಲ',
        'Malayalam': 'സുഖമായി തോന്നുന്നു, പ്രശ്നമൊന്നുമില്ല',
        'Punjabi': 'ਠੀਕ ਮਹਿਸੂਸ ਕਰ ਰਿਹਾ/ਰਹੀ ਹਾਂ, ਕੋਈ ਸਮੱਸਿਆ ਨਹੀਂ',
        'Odia': 'ଭଲ ଲାଗୁଛି, କୌଣସି ସମସ୍ୟା ନାହିଁ',
        'Assamese': 'ভাল অনুভৱ কৰিছোঁ, কোনো সমস্যা নাই',
        'Urdu': 'ٹھیک محسوس کر رہا/رہی ہوں، کوئی مسئلہ نہیں'
    },
    'some_concerns': {
        'English': 'Some minor concerns',
        'Hindi': 'कुछ छोटी चिंताएं हैं',
        'Telugu': 'కొన్ని చిన్న ఆందోళనలు ఉన్నాయి',
        'Bengali': 'কিছু ছোটখাটো উদ্বেগ আছে',
        'Marathi': 'काही लहान चिंता आहेत',
        'Tamil': 'சில சிறிய கவலைகள் உள்ளன',
        'Gujarati': 'કેટલીક નાની ચિંતાઓ છે',
        'Kannada': 'ಕೆಲವು ಸಣ್ಣ ಕಾಳಜಿಗಳಿವೆ',
        'Malayalam': 'ചില ചെറിയ ആശങ്കകൾ ഉണ്ട്',
        'Punjabi': 'ਕੁਝ ਛੋਟੀਆਂ ਚਿੰਤਾਵਾਂ ਹਨ',
        'Odia': 'କିଛି ଛୋଟ ଚିନ୍ତା ଅଛି',
        'Assamese': 'কিছু সৰু চিন্তা আছে',
        'Urdu': 'کچھ چھوٹی پریشانیاں ہیں'
    },
    'side_effects': {
        'English': 'Experiencing side effects',
        'Hindi': 'साइड इफेक्ट्स महसूस हो रहे हैं',
        'Telugu': 'సైడ్ ఎఫెక్ట్స్ అనుభవిస్తున్నాను',
        'Bengali': 'পার্শ্বপ্রতিক্রিয়া অনুভব করছি',
        'Marathi': 'साइड इफेक्ट्स जाणवत आहेत',
        'Tamil': 'பக்க விளைவுகள் உணர்கிறேன்',
        'Gujarati': 'સાઇડ ઇફેક્ટ્સ અનુભવી રહ્યો/રહી છું',
        'Kannada': 'ಅಡ್ಡಪರಿಣಾಮಗಳು ಅನುಭವಿಸುತ್ತಿದ್ದೇನೆ',
        'Malayalam': 'പാർശ്വഫലങ്ങൾ അനുഭവിക്കുന്നു',
        'Punjabi': 'ਸਾਈਡ ਇਫੈਕਟਸ ਮਹਿਸੂਸ ਹੋ ਰਹੇ ਹਨ',
        'Odia': 'ପାର୍ଶ୍ୱ ପ୍ରତିକ୍ରିୟା ଅନୁଭବ ହେଉଛି',
        'Assamese': 'পাৰ্শ্বক্ৰিয়া অনুভৱ হৈছে',
        'Urdu': 'سائیڈ ایفیکٹس محسوس ہو رہے ہیں'
    },
    'symptoms_worse': {
        'English': 'Symptoms are getting worse',
        'Hindi': 'लक्षण बिगड़ रहे हैं',
        'Telugu': 'లక్షణాలు మరింత తీవ్రమవుతున్నాయి',
        'Bengali': 'লক্ষণগুলো আরও খারাপ হচ্ছে',
        'Marathi': 'लक्षणे वाढत आहेत',
        'Tamil': 'அறிகுறிகள் மோசமாகிறது',
        'Gujarati': 'લક્ષણો વધુ ખરાબ થઈ રહ્યા છે',
        'Kannada': 'ರೋಗಲಕ್ಷಣಗಳು ಹದಗೆಡುತ್ತಿವೆ',
        'Malayalam': 'ലക്ഷണങ്ങൾ വഷളാവുകയാണ്',
        'Punjabi': 'ਲੱਛਣ ਹੋਰ ਮਾੜੇ ਹੋ ਰਹੇ ਹਨ',
        'Odia': 'ଲକ୍ଷଣ ଆହୁରି ଖରାପ ହେଉଛି',
        'Assamese': 'লক্ষণ বেয়া হৈ গৈ আছে',
        'Urdu': 'علامات مزید خراب ہو رہی ہیں'
    },
    'need_help': {
        'English': 'Need medical help',
        'Hindi': 'चिकित्सा सहायता चाहिए',
        'Telugu': 'వైద్య సహాయం అవసరం',
        'Bengali': 'চিকিৎসা সহায়তা প্রয়োজন',
        'Marathi': 'वैद्यकीय मदत हवी आहे',
        'Tamil': 'மருத்துவ உதவி தேவை',
        'Gujarati': 'તબીબી સહાય જોઈએ છે',
        'Kannada': 'ವೈದ್ಯಕೀಯ ಸಹಾಯ ಬೇಕು',
        'Malayalam': 'വൈദ്യസഹായം ആവശ്യമാണ്',
        'Punjabi': 'ਡਾਕਟਰੀ ਮਦਦ ਚਾਹੀਦੀ ਹੈ',
        'Odia': 'ଡାକ୍ତରୀ ସାହାଯ୍ୟ ଦରକାର',
        'Assamese': 'চিকিৎসা সহায়তা লাগে',
        'Urdu': 'طبی مدد چاہیے'
    },
    'yes': {
        'English': 'Yes',
        'Hindi': 'हाँ',
        'Telugu': 'అవును',
        'Bengali': 'হ্যাঁ',
        'Marathi': 'होय',
        'Tamil': 'ஆம்',
        'Gujarati': 'હા',
        'Kannada': 'ಹೌದು',
        'Malayalam': 'അതെ',
        'Punjabi': 'ਹਾਂ',
        'Odia': 'ହଁ',
        'Assamese': 'হয়',
        'Urdu': 'ہاں'
    },
    'no': {
        'English': 'No',
        'Hindi': 'नहीं',
        'Telugu': 'లేదు',
        'Bengali': 'না',
        'Marathi': 'नाही',
        'Tamil': 'இல்லை',
        'Gujarati': 'ના',
        'Kannada': 'ಇಲ್ಲ',
        'Malayalam': 'ഇല്ല',
        'Punjabi': 'ਨਹੀਂ',
        'Odia': 'ନା',
        'Assamese': 'নাই',
        'Urdu': 'نہیں'
    },
    'not_sure': {
        'English': 'Not sure',
        'Hindi': 'पता नहीं',
        'Telugu': 'తెలియదు',
        'Bengali': 'নিশ্চিত নই',
        'Marathi': 'माहित नाही',
        'Tamil': 'தெரியவில்லை',
        'Gujarati': 'ખાતરી નથી',
        'Kannada': 'ಗೊತ್ತಿಲ್ಲ',
        'Malayalam': 'ഉറപ്പില്ല',
        'Punjabi': 'ਪਤਾ ਨਹੀਂ',
        'Odia': 'ଜଣା ନାହିଁ',
        'Assamese': 'নিশ্চিত নহয়',
        'Urdu': 'پتا نہیں'
    },
    # Day 3 options
    'better': {
        'English': 'Better',
        'Hindi': 'बेहतर',
        'Telugu': 'మెరుగ్గా',
        'Bengali': 'ভালো',
        'Marathi': 'चांगले',
        'Tamil': 'சிறந்தது',
        'Gujarati': 'સારું',
        'Kannada': 'ಉತ್ತಮ',
        'Malayalam': 'മികച്ചത്',
        'Punjabi': 'ਬਿਹਤਰ',
        'Odia': 'ଭଲ',
        'Assamese': 'ভাল',
        'Urdu': 'بہتر'
    },
    'same': {
        'English': 'Same as before',
        'Hindi': 'पहले जैसा ही',
        'Telugu': 'ఇంతకు ముందు లాగే',
        'Bengali': 'আগের মতোই',
        'Marathi': 'आधीसारखेच',
        'Tamil': 'முன்பு போலவே',
        'Gujarati': 'પહેલા જેવું જ',
        'Kannada': 'ಮೊದಲಿನಂತೆ',
        'Malayalam': 'മുമ്പത്തെ പോലെ',
        'Punjabi': 'ਪਹਿਲਾਂ ਵਰਗਾ',
        'Odia': 'ପୂର୍ବ ପରି',
        'Assamese': 'আগৰ দৰেই',
        'Urdu': 'پہلے جیسا'
    },
    'worse': {
        'English': 'Worse',
        'Hindi': 'खराब',
        'Telugu': 'అధ్వాన్నంగా',
        'Bengali': 'খারাপ',
        'Marathi': 'वाईट',
        'Tamil': 'மோசமானது',
        'Gujarati': 'ખરાબ',
        'Kannada': 'ಕೆಟ್ಟದು',
        'Malayalam': 'മോശം',
        'Punjabi': 'ਮਾੜਾ',
        'Odia': 'ଖରାପ',
        'Assamese': 'বেয়া',
        'Urdu': 'برا'
    },
    'resolved': {
        'English': 'Completely resolved',
        'Hindi': 'पूरी तरह ठीक',
        'Telugu': 'పూర్తిగా పరిష్కారమైంది',
        'Bengali': 'সম্পূর্ণ সমাধান',
        'Marathi': 'पूर्णपणे बरे',
        'Tamil': 'முழுமையாக தீர்ந்தது',
        'Gujarati': 'સંપૂર્ણ ઠીક',
        'Kannada': 'ಸಂಪೂರ್ಣ ಗುಣ',
        'Malayalam': 'പൂർണ്ണമായി ഭേദമായി',
        'Punjabi': 'ਪੂਰੀ ਤਰ੍ਹਾਂ ਠੀਕ',
        'Odia': 'ସମ୍ପୂର୍ଣ୍ଣ ଭଲ',
        'Assamese': 'সম্পূৰ্ণ ভাল',
        'Urdu': 'مکمل طور پر ٹھیک'
    },
    'yes_consulted': {
        'English': 'Yes, I have consulted a doctor',
        'Hindi': 'हाँ, मैंने डॉक्टर से परामर्श किया',
        'Telugu': 'అవును, నేను డాక్టర్‌ను సంప్రదించాను',
        'Bengali': 'হ্যাঁ, ডাক্তারের সাথে পরামর্শ করেছি',
        'Marathi': 'होय, मी डॉक्टरांचा सल्ला घेतला',
        'Tamil': 'ஆம், மருத்துவரை அணுகினேன்',
        'Gujarati': 'હા, ડોક્ટરની સલાહ લીધી',
        'Kannada': 'ಹೌದು, ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿದ್ದೇನೆ',
        'Malayalam': 'അതെ, ഡോക്ടറെ കണ്ടു',
        'Punjabi': 'ਹਾਂ, ਡਾਕਟਰ ਨਾਲ ਸਲਾਹ ਕੀਤੀ',
        'Odia': 'ହଁ, ଡାକ୍ତରଙ୍କ ସହ ପରାମର୍ଶ କରିଛି',
        'Assamese': 'হয়, ডাক্তৰৰ লগত আলোচনা কৰিছোঁ',
        'Urdu': 'ہاں، ڈاکٹر سے مشورہ کیا'
    },
    'plan_to': {
        'English': 'No, but I plan to',
        'Hindi': 'नहीं, लेकिन मैं करने वाला हूँ',
        'Telugu': 'లేదు, కానీ చేయాలని అనుకుంటున్నాను',
        'Bengali': 'না, কিন্তু করব',
        'Marathi': 'नाही, पण करणार आहे',
        'Tamil': 'இல்லை, ஆனால் செய்யப் போகிறேன்',
        'Gujarati': 'ના, પણ કરીશ',
        'Kannada': 'ಇಲ್ಲ, ಆದರೆ ಮಾಡುತ್ತೇನೆ',
        'Malayalam': 'ഇല്ല, പക്ഷേ ചെയ്യാം',
        'Punjabi': 'ਨਹੀਂ, ਪਰ ਕਰਾਂਗਾ',
        'Odia': 'ନା, କିନ୍ତୁ କରିବି',
        'Assamese': 'নাই, কিন্তু কৰিম',
        'Urdu': 'نہیں، لیکن کروں گا'
    },
    'not_necessary': {
        'English': 'I don\'t think it\'s necessary',
        'Hindi': 'मुझे नहीं लगता जरूरी है',
        'Telugu': 'అవసరం లేదని అనుకుంటున్నాను',
        'Bengali': 'প্রয়োজন মনে হয় না',
        'Marathi': 'गरज नाही असे वाटते',
        'Tamil': 'தேவையில்லை என்று நினைக்கிறேன்',
        'Gujarati': 'જરૂરી નથી લાગતું',
        'Kannada': 'ಅಗತ್ಯವಿಲ್ಲ ಎಂದು ತೋರುತ್ತದೆ',
        'Malayalam': 'ആവശ്യമില്ല എന്ന് തോന്നുന്നു',
        'Punjabi': 'ਜ਼ਰੂਰੀ ਨਹੀਂ ਲੱਗਦਾ',
        'Odia': 'ଆବଶ୍ୟକ ନାହିଁ ଲାଗୁଛି',
        'Assamese': 'প্ৰয়োজন নাই বুলি ভাবোঁ',
        'Urdu': 'ضرورت نہیں لگتی'
    },
    'continuing': {
        'English': 'Yes, continuing the medication',
        'Hindi': 'हाँ, दवाई जारी है',
        'Telugu': 'అవును, మందు కొనసాగిస్తున్నాను',
        'Bengali': 'হ্যাঁ, ওষুধ চালু আছে',
        'Marathi': 'होय, औषध चालू आहे',
        'Tamil': 'ஆம், மருந்து தொடர்கிறேன்',
        'Gujarati': 'હા, દવા ચાલુ છે',
        'Kannada': 'ಹೌದು, ಔಷಧಿ ಮುಂದುವರಿದಿದೆ',
        'Malayalam': 'അതെ, മരുന്ന് തുടരുന്നു',
        'Punjabi': 'ਹਾਂ, ਦਵਾਈ ਜਾਰੀ ਹੈ',
        'Odia': 'ହଁ, ଔଷଧ ଜାରି ଅଛି',
        'Assamese': 'হয়, ঔষধ চলি আছে',
        'Urdu': 'ہاں، دوا جاری ہے'
    },
    'stopped_self': {
        'English': 'Stopped on my own',
        'Hindi': 'खुद बंद कर दी',
        'Telugu': 'నా అంతట నేను ఆపేసాను',
        'Bengali': 'নিজে বন্ধ করেছি',
        'Marathi': 'स्वतः थांबवले',
        'Tamil': 'சுயமாக நிறுத்தினேன்',
        'Gujarati': 'જાતે બંધ કર્યું',
        'Kannada': 'ಸ್ವಯಂ ನಿಲ್ಲಿಸಿದೆ',
        'Malayalam': 'സ്വയം നിർത്തി',
        'Punjabi': 'ਖੁਦ ਬੰਦ ਕੀਤੀ',
        'Odia': 'ନିଜେ ବନ୍ଦ କରିଦେଲି',
        'Assamese': 'নিজে বন্ধ কৰিলোঁ',
        'Urdu': 'خود بند کر دی'
    },
    'doctor_stop': {
        'English': 'Doctor advised to stop',
        'Hindi': 'डॉक्टर ने बंद करने को कहा',
        'Telugu': 'డాక్టర్ ఆపమని చెప్పారు',
        'Bengali': 'ডাক্তার বন্ধ করতে বললেন',
        'Marathi': 'डॉक्टरांनी थांबवायला सांगितले',
        'Tamil': 'மருத்துவர் நிறுத்தச் சொன்னார்',
        'Gujarati': 'ડોક્ટરે બંધ કરવા કહ્યું',
        'Kannada': 'ವೈದ್ಯರು ನಿಲ್ಲಿಸಲು ಹೇಳಿದರು',
        'Malayalam': 'ഡോക്ടർ നിർത്താൻ പറഞ്ഞു',
        'Punjabi': 'ਡਾਕਟਰ ਨੇ ਬੰਦ ਕਰਨ ਲਈ ਕਿਹਾ',
        'Odia': 'ଡାକ୍ତର ବନ୍ଦ କରିବାକୁ କହିଲେ',
        'Assamese': 'ডাক্তৰে বন্ধ কৰিবলৈ কৈছে',
        'Urdu': 'ڈاکٹر نے بند کرنے کو کہا'
    },
    'changed_dose': {
        'English': 'Changed dosage',
        'Hindi': 'खुराक बदल दी',
        'Telugu': 'డోసేజ్ మార్చారు',
        'Bengali': 'ডোজ বদলেছি',
        'Marathi': 'डोस बदलला',
        'Tamil': 'டோஸ் மாற்றினேன்',
        'Gujarati': 'ડોઝ બદલ્યો',
        'Kannada': 'ಡೋಸ್ ಬದಲಾಯಿಸಿದೆ',
        'Malayalam': 'ഡോസ് മാറ്റി',
        'Punjabi': 'ਡੋਜ਼ ਬਦਲੀ',
        'Odia': 'ଡୋଜ୍ ବଦଳାଇଲି',
        'Assamese': "ড'জ সলনি কৰিলোঁ",
        'Urdu': 'خوراک بدل دی'
    },
    'yes_new': {
        'English': 'Yes, I have new symptoms',
        'Hindi': 'हाँ, नए लक्षण हैं',
        'Telugu': 'అవును, కొత్త లక్షణాలు ఉన్నాయి',
        'Bengali': 'হ্যাঁ, নতুন লক্ষণ আছে',
        'Marathi': 'होय, नवीन लक्षणे आहेत',
        'Tamil': 'ஆம், புதிய அறிகுறிகள் உள்ளன',
        'Gujarati': 'હા, નવા લક્ષણો છે',
        'Kannada': 'ಹೌದು, ಹೊಸ ಲಕ್ಷಣಗಳಿವೆ',
        'Malayalam': 'അതെ, പുതിയ ലക്ഷണങ്ങൾ ഉണ്ട്',
        'Punjabi': 'ਹਾਂ, ਨਵੇਂ ਲੱਛਣ ਹਨ',
        'Odia': 'ହଁ, ନୂଆ ଲକ୍ଷଣ ଅଛି',
        'Assamese': 'হয়, নতুন লক্ষণ আছে',
        'Urdu': 'ہاں، نئی علامات ہیں'
    },
    'no_new': {
        'English': 'No new symptoms',
        'Hindi': 'कोई नया लक्षण नहीं',
        'Telugu': 'కొత్త లక్షణాలు లేవు',
        'Bengali': 'নতুন কোনো লক্ষণ নেই',
        'Marathi': 'नवीन लक्षणे नाहीत',
        'Tamil': 'புதிய அறிகுறிகள் இல்லை',
        'Gujarati': 'નવા લક્ષણો નથી',
        'Kannada': 'ಹೊಸ ಲಕ್ಷಣಗಳಿಲ್ಲ',
        'Malayalam': 'പുതിയ ലക്ഷണങ്ങൾ ഇല്ല',
        'Punjabi': 'ਕੋਈ ਨਵਾਂ ਲੱਛਣ ਨਹੀਂ',
        'Odia': 'କୌଣସି ନୂଆ ଲକ୍ଷଣ ନାହିଁ',
        'Assamese': 'কোনো নতুন লক্ষণ নাই',
        'Urdu': 'کوئی نئی علامت نہیں'
    },
    'improving': {
        'English': 'Improving',
        'Hindi': 'सुधार हो रहा है',
        'Telugu': 'మెరుగుపడుతోంది',
        'Bengali': 'উন্নতি হচ্ছে',
        'Marathi': 'सुधारत आहे',
        'Tamil': 'மேம்படுகிறது',
        'Gujarati': 'સુધારો થઈ રહ્યો છે',
        'Kannada': 'ಸುಧಾರಿಸುತ್ತಿದೆ',
        'Malayalam': 'മെച്ചപ്പെടുന്നു',
        'Punjabi': 'ਸੁਧਾਰ ਹੋ ਰਿਹਾ ਹੈ',
        'Odia': 'ଉନ୍ନତି ହେଉଛି',
        'Assamese': 'উন্নতি হৈ আছে',
        'Urdu': 'بہتری ہو رہی ہے'
    },
    'fully_recovered': {
        'English': 'Fully recovered',
        'Hindi': 'पूरी तरह ठीक हो गया',
        'Telugu': 'పూర్తిగా కోలుకున్నాను',
        'Bengali': 'সম্পূর্ণ সুস্থ',
        'Marathi': 'पूर्ण बरा झालो',
        'Tamil': 'முழுமையாக குணமானேன்',
        'Gujarati': 'સંપૂર્ણ સ્વસ્થ',
        'Kannada': 'ಸಂಪೂರ್ಣ ಚೇತರಿಕೆ',
        'Malayalam': 'പൂർണ്ണമായി സുഖപ്പെട്ടു',
        'Punjabi': 'ਪੂਰੀ ਤਰ੍ਹਾਂ ਠੀਕ',
        'Odia': 'ସମ୍ପୂର୍ଣ୍ଣ ସୁସ୍ଥ',
        'Assamese': 'সম্পূৰ্ণ সুস্থ',
        'Urdu': 'مکمل صحتیاب'
    },
    'mostly_recovered': {
        'English': 'Mostly recovered',
        'Hindi': 'लगभग ठीक हो गया',
        'Telugu': 'చాలావరకు కోలుకున్నాను',
        'Bengali': 'প্রায় সুস্থ',
        'Marathi': 'जवळपास बरा झालो',
        'Tamil': 'பெரும்பாலும் குணமானேன்',
        'Gujarati': 'મોટાભાગે સ્વસ્થ',
        'Kannada': 'ಹೆಚ್ಚಾಗಿ ಚೇತರಿಕೆ',
        'Malayalam': 'ഏതാണ്ട് സുഖപ്പെട്ടു',
        'Punjabi': 'ਲਗਭਗ ਠੀਕ',
        'Odia': 'ପ୍ରାୟ ସୁସ୍ଥ',
        'Assamese': 'প্ৰায় সুস্থ',
        'Urdu': 'تقریباً صحتیاب'
    },
    'still_symptoms': {
        'English': 'Still experiencing symptoms',
        'Hindi': 'अभी भी लक्षण हैं',
        'Telugu': 'ఇంకా లక్షణాలు ఉన్నాయి',
        'Bengali': 'এখনও লক্ষণ আছে',
        'Marathi': 'अजूनही लक्षणे आहेत',
        'Tamil': 'இன்னும் அறிகுறிகள் உள்ளன',
        'Gujarati': 'હજુ લક્ષણો છે',
        'Kannada': 'ಇನ್ನೂ ಲಕ್ಷಣಗಳಿವೆ',
        'Malayalam': 'ഇപ്പോഴും ലക്ഷണങ്ങൾ ഉണ്ട്',
        'Punjabi': 'ਅਜੇ ਵੀ ਲੱਛਣ ਹਨ',
        'Odia': 'ଏବେ ବି ଲକ୍ଷଣ ଅଛି',
        'Assamese': 'এতিয়াও লক্ষণ আছে',
        'Urdu': 'ابھی بھی علامات ہیں'
    },
    'worsened': {
        'English': 'Symptoms have worsened',
        'Hindi': 'लक्षण बिगड़ गए हैं',
        'Telugu': 'లక్షణాలు మరింత తీవ్రమయ్యాయి',
        'Bengali': 'লক্ষণ আরও খারাপ হয়েছে',
        'Marathi': 'लक्षणे वाढली आहेत',
        'Tamil': 'அறிகுறிகள் மோசமாகிவிட்டன',
        'Gujarati': 'લક્ષણો વધુ ખરાબ થયા છે',
        'Kannada': 'ಲಕ್ಷಣಗಳು ಹದಗೆಟ್ಟಿವೆ',
        'Malayalam': 'ലക്ഷണങ്ങൾ വഷളായി',
        'Punjabi': 'ਲੱਛਣ ਹੋਰ ਮਾੜੇ ਹੋ ਗਏ',
        'Odia': 'ଲକ୍ଷଣ ଆହୁରି ଖରାପ ହୋଇଛି',
        'Assamese': 'লক্ষণ বেয়া হৈ গৈছে',
        'Urdu': 'علامات مزید خراب ہو گئیں'
    },
    'yes_arrange': {
        'English': 'Yes, please arrange a check-up',
        'Hindi': 'हाँ, कृपया चेक-अप की व्यवस्था करें',
        'Telugu': 'అవును, దయచేసి చెక్-అప్ ఏర్పాటు చేయండి',
        'Bengali': 'হ্যাঁ, চেক-আপের ব্যবস্থা করুন',
        'Marathi': 'होय, कृपया तपासणी करा',
        'Tamil': 'ஆம், சோதனை ஏற்பாடு செய்யுங்கள்',
        'Gujarati': 'હા, ચેક-અપ ગોઠવો',
        'Kannada': 'ಹೌದು, ಚೆಕ್-ಅಪ್ ಏರ್ಪಡಿಸಿ',
        'Malayalam': 'അതെ, ചെക്ക്-അപ്പ് ക്രമീകരിക്കുക',
        'Punjabi': 'ਹਾਂ, ਚੈੱਕ-ਅੱਪ ਕਰਵਾਓ',
        'Odia': 'ହଁ, ଚେକ୍-ଅପ୍ କରନ୍ତୁ',
        'Assamese': 'হয়, চেক-আপ কৰাওক',
        'Urdu': 'ہاں، چیک اپ کا بندوبست کریں'
    },
    'no_thanks': {
        'English': 'No, thank you',
        'Hindi': 'नहीं, धन्यवाद',
        'Telugu': 'లేదు, ధన్యవాదాలు',
        'Bengali': 'না, ধন্যবাদ',
        'Marathi': 'नाही, धन्यवाद',
        'Tamil': 'வேண்டாம், நன்றி',
        'Gujarati': 'ના, આભાર',
        'Kannada': 'ಇಲ್ಲ, ಧನ್ಯವಾದ',
        'Malayalam': 'വേണ്ട, നന്ദി',
        'Punjabi': 'ਨਹੀਂ, ਧੰਨਵਾਦ',
        'Odia': 'ନା, ଧନ୍ୟବାଦ',
        'Assamese': 'নালাগে, ধন্যবাদ',
        'Urdu': 'نہیں، شکریہ'
    },
    'maybe_later': {
        'English': 'Maybe later',
        'Hindi': 'शायद बाद में',
        'Telugu': 'తర్వాత చూద్దాం',
        'Bengali': 'পরে দেখা যাবে',
        'Marathi': 'नंतर बघू',
        'Tamil': 'பின்னால் பார்க்கலாம்',
        'Gujarati': 'પછી જોઈશું',
        'Kannada': 'ನಂತರ ನೋಡೋಣ',
        'Malayalam': 'പിന്നീട് കാണാം',
        'Punjabi': 'ਬਾਅਦ ਵਿੱਚ ਦੇਖਾਂਗੇ',
        'Odia': 'ପରେ ଦେଖିବା',
        'Assamese': 'পিছত চাম',
        'Urdu': 'بعد میں دیکھیں گے'
    }
}

# Standard options for each question type
QUESTION_OPTIONS = {
    # Day 1
    'day1_q1': ['feeling_fine', 'some_concerns', 'side_effects', 'symptoms_worse', 'need_help'],
    'day1_q2': ['yes', 'no', 'not_sure'],
    'day1_q3': None,  # Numeric 1-10
    'day1_q4': None,  # Free text date
    # Day 3
    'day3_q1': ['better', 'same', 'worse', 'resolved'],
    'day3_q2': ['yes_consulted', 'plan_to', 'not_necessary'],
    'day3_q3': ['continuing', 'stopped_self', 'doctor_stop', 'changed_dose'],
    'day3_q4': ['yes_new', 'no_new'],
    # Day 5
    'day5_q1': ['improving', 'same', 'worse'],
    'day5_q2': ['yes', 'no'],
    'day5_q3': None,  # Free text
    'day5_q4': None,  # Free text
    # Day 7
    'day7_q1': ['fully_recovered', 'mostly_recovered', 'still_symptoms', 'worsened'],
    'day7_q2': None,  # Date
    'day7_q3': ['yes_arrange', 'no_thanks', 'maybe_later'],
    'day7_q4': None,  # Free text
}


def get_translated_question(question_id: str, language: str) -> str:
    """Get question text in the specified language."""
    if question_id in QUESTION_TRANSLATIONS:
        return QUESTION_TRANSLATIONS[question_id].get(language, QUESTION_TRANSLATIONS[question_id]['English'])
    return None


def get_translated_options(question_id: str, language: str) -> List[Dict]:
    """Get translated options for a question with numbered choices."""
    option_keys = QUESTION_OPTIONS.get(question_id)
    if not option_keys:
        return None
    
    options = []
    for i, key in enumerate(option_keys, 1):
        if key in OPTION_TRANSLATIONS:
            translated = OPTION_TRANSLATIONS[key].get(language, OPTION_TRANSLATIONS[key]['English'])
            options.append({
                'number': i,
                'key': key,
                'text': translated
            })
    return options


def format_question_with_options(question_id: str, question_text: str, language: str) -> str:
    """Format question with numbered options in the selected language."""
    options = get_translated_options(question_id, language)
    
    if not options:
        # For questions without predefined options (like severity scale or date)
        if 'q3' in question_id:  # Severity question
            scale_text = {
                'English': '(1 = very mild, 10 = very severe)',
                'Hindi': '(1 = बहुत हल्का, 10 = बहुत गंभीर)',
                'Telugu': '(1 = చాలా తేలికగా, 10 = చాలా తీవ్రంగా)',
                'Bengali': '(1 = খুব হালকা, 10 = খুব গুরুতর)',
                'Marathi': '(1 = खूप सौम्य, 10 = खूप तीव्र)',
                'Tamil': '(1 = மிகவும் லேசான, 10 = மிகவும் கடுமையான)',
                'Gujarati': '(1 = ખૂબ હળવું, 10 = ખૂબ ગંભીર)',
                'Kannada': '(1 = ಅತಿ ಸೌಮ್ಯ, 10 = ಅತಿ ತೀವ್ರ)',
                'Malayalam': '(1 = വളരെ സൗമ്യം, 10 = വളരെ കഠിനം)',
                'Punjabi': '(1 = ਬਹੁਤ ਹਲਕਾ, 10 = ਬਹੁਤ ਗੰਭੀਰ)',
                'Odia': '(1 = ବହୁତ ହାଲୁକା, 10 = ବହୁତ ଗୁରୁତର)',
                'Assamese': '(1 = অতি সাধাৰণ, 10 = অতি গুৰুতৰ)',
                'Urdu': '(1 = بہت ہلکا، 10 = بہت شدید)'
            }
            return f"{question_text}\n\n{scale_text.get(language, scale_text['English'])}\n\n_Reply with a number from 1 to 10_"
        return question_text
    
    # Format options as numbered list
    formatted_options = "\n".join([f"{opt['number']}️⃣ {opt['text']}" for opt in options])
    
    reply_text = {
        'English': '_Reply with the number (1-{count})_',
        'Hindi': '_नंबर से जवाब दें (1-{count})_',
        'Telugu': '_సంఖ్యతో జవాబు ఇవ్వండి (1-{count})_',
        'Bengali': '_নম্বর দিয়ে উত্তর দিন (1-{count})_',
        'Marathi': '_क्रमांकाने उत्तर द्या (1-{count})_',
        'Tamil': '_எண்ணுடன் பதிலளிக்கவும் (1-{count})_',
        'Gujarati': '_નંબરથી જવાબ આપો (1-{count})_',
        'Kannada': '_ಸಂಖ್ಯೆಯೊಂದಿಗೆ ಉತ್ತರಿಸಿ (1-{count})_',
        'Malayalam': '_നമ്പർ ഉപയോഗിച്ച് മറുപടി നൽകുക (1-{count})_',
        'Punjabi': '_ਨੰਬਰ ਨਾਲ ਜਵਾਬ ਦਿਓ (1-{count})_',
        'Odia': '_ନମ୍ବର ସହ ଉତ୍ତର ଦିଅନ୍ତୁ (1-{count})_',
        'Assamese': '_নম্বৰেৰে উত্তৰ দিয়ক (1-{count})_',
        'Urdu': '_نمبر سے جواب دیں (1-{count})_'
    }
    
    reply_instruction = reply_text.get(language, reply_text['English']).format(count=len(options))
    
    return f"{question_text}\n\n{formatted_options}\n\n{reply_instruction}"


DAY_WISE_PREDEFINED_QUESTIONS = {
    1: [
        # Day 1: Initial wellness check and symptom confirmation
        {
            'id': 'day1_q1',
            'question': 'How are you feeling today after taking the medication?',
            'maps_to_column': None,
            'purpose': 'wellness_check',
            'options': ['I am feeling fine', 'I have some concerns', 'I am experiencing side effects']
        },
        {
            'id': 'day1_q2',
            'question': 'Are you still experiencing the symptoms you reported earlier?',
            'maps_to_column': 'symptoms',
            'purpose': 'symptom_update'
        },
        {
            'id': 'day1_q3',
            'question': 'On a scale of 1-10, how would you rate the severity of your symptoms? (1 = very mild, 10 = very severe)',
            'maps_to_column': 'risk_level',
            'purpose': 'severity_assessment'
        },
        {
            'id': 'day1_q4',
            'question': 'When did you first notice these symptoms? (Please share the approximate date)',
            'maps_to_column': 'symptom_onset_date',
            'purpose': 'temporal_clarity'
        }
    ],
    3: [
        # Day 3: Symptom progression and medical consultation
        {
            'id': 'day3_q1',
            'question': 'How have your symptoms changed since we last spoke? (Better, Same, or Worse)',
            'maps_to_column': 'symptoms',
            'purpose': 'symptom_progression',
            'options': ['Better', 'Same', 'Worse', 'Completely resolved']
        },
        {
            'id': 'day3_q2',
            'question': 'Have you consulted a doctor or healthcare provider about these symptoms?',
            'maps_to_column': 'doctor_confirmed',
            'purpose': 'medical_confirmation',
            'options': ['Yes, I have consulted', 'No, but I plan to', 'No, I don\'t think it\'s necessary']
        },
        {
            'id': 'day3_q3',
            'question': 'Are you still taking the medication that caused these symptoms?',
            'maps_to_column': None,
            'purpose': 'medication_status',
            'options': ['Yes, continuing', 'Stopped on my own', 'Doctor advised to stop', 'Changed dosage']
        },
        {
            'id': 'day3_q4',
            'question': 'Have you noticed any new symptoms since we last contacted you?',
            'maps_to_column': 'symptoms',
            'purpose': 'new_symptoms'
        }
    ],
    5: [
        # Day 5: Detailed clinical data and impact assessment
        {
            'id': 'day5_q1',
            'question': 'Are your symptoms improving, staying the same, or getting worse?',
            'maps_to_column': 'symptoms',
            'purpose': 'symptom_trend',
            'options': ['Improving', 'Staying the same', 'Getting worse']
        },
        {
            'id': 'day5_q2',
            'question': 'Did you need to visit a hospital or emergency room due to these symptoms?',
            'maps_to_column': 'hospital_confirmed',
            'purpose': 'hospital_confirmation',
            'options': ['Yes', 'No']
        },
        {
            'id': 'day5_q3',
            'question': 'How have these symptoms affected your daily activities? (work, sleep, eating)',
            'maps_to_column': None,
            'purpose': 'impact_assessment'
        },
        {
            'id': 'day5_q4',
            'question': 'Have you taken any other medications or treatments to manage these symptoms?',
            'maps_to_column': None,
            'purpose': 'treatment_actions'
        }
    ],
    7: [
        # Day 7: Resolution and final assessment
        {
            'id': 'day7_q1',
            'question': 'What is the current status of your symptoms?',
            'maps_to_column': 'symptoms',
            'purpose': 'final_status',
            'options': ['Fully recovered', 'Mostly recovered', 'Still experiencing symptoms', 'Symptoms worsened']
        },
        {
            'id': 'day7_q2',
            'question': 'If your symptoms have resolved, when did they stop? (Please share the approximate date)',
            'maps_to_column': 'symptom_resolution_date',
            'purpose': 'resolution_date'
        },
        {
            'id': 'day7_q3',
            'question': 'Would you like us to arrange a free health check-up from the pharmaceutical company?',
            'maps_to_column': None,
            'purpose': 'pharma_recall_offer',
            'options': ['Yes, please arrange', 'No, thank you', 'Maybe later']
        },
        {
            'id': 'day7_q4',
            'question': 'Is there anything else about your experience with this medication that you would like to share?',
            'maps_to_column': 'symptoms',
            'purpose': 'final_feedback'
        }
    ]
}

# Legacy support - default predefined questions (Day 1)
PREDEFINED_QUESTIONS = DAY_WISE_PREDEFINED_QUESTIONS[1]


def get_day_specific_questions(day: int) -> List[Dict]:
    """
    Get predefined questions for a specific day in the follow-up cycle.
    
    Args:
        day: The follow-up day (1, 3, 5, or 7)
        
    Returns:
        List of predefined question dictionaries for that day
    """
    return DAY_WISE_PREDEFINED_QUESTIONS.get(day, DAY_WISE_PREDEFINED_QUESTIONS[1])


def get_combined_questions(patient, previous_responses: Dict = None, current_day: int = 1, language: str = "English") -> Dict[str, Any]:
    """
    Get combined predefined + LLM questions for a patient for a specific day.
    
    Args:
        patient: Patient model object
        previous_responses: Dict of previous day's responses  
        current_day: The current follow-up day (1, 3, 5, or 7)
        language: Patient's preferred language (e.g., 'Telugu', 'Hindi', 'English')
    
    Returns:
        Dict with predefined_questions, llm_questions, and all_questions
    """
    service = PrivacySafeLLMService()
    
    # Get day-specific predefined questions
    predefined = get_day_specific_questions(current_day)
    
    # Get personalized LLM questions based on case scoring and day
    print(f"🌐 Getting personalized questions in {language} for Day {current_day}...")
    llm_result = service.get_personalized_day_questions(patient, previous_responses, current_day, language)
    
    # Combine all questions
    all_questions = predefined + llm_result.get('suggested_questions', [])
    
    return {
        'current_day': current_day,
        'predefined_questions': predefined,
        'llm_questions': llm_result.get('suggested_questions', []),
        'all_questions': all_questions,
        'analysis': llm_result.get('analysis', ''),
        'priority': llm_result.get('priority', 'medium'),
        'focus_areas': llm_result.get('focus_areas', []),
        'llm_provider': service.llm_provider,
        'language': language
    }
