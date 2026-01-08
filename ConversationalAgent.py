"""
📋 CONVERSATIONAL AGENT FOR PHARMACOVIGILANCE WHATSAPP FOLLOW-UP

This module implements Agent 1: a visit-triggered, WhatsApp-based pharmacovigilance
follow-up system driven by a hospital database.

CORE PRINCIPLES (NON-NEGOTIABLE):
1. Each conversation corresponds to exactly ONE visit_id
2. Patient identity already exists; do not create or modify patient records
3. Never overwrite past answers
4. Never infer logic from WhatsApp message history
5. WhatsApp is UI only; the agent controls state

SAFETY RULES:
- Do NOT give medical advice
- Do NOT diagnose
- Always run safety checks before ending
- If severe symptoms appear: mark safety flag, do NOT silently end conversation

Author: Converted from firstAgentTest.ipynb
"""

# =============================================================================
# IMPORTS
# =============================================================================

from typing import TypedDict, Literal, Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json
import re

# -----------------------------------------------------------------------------
# NOTE: LangGraph is used for state machine flow control
# Import will fail if langgraph is not installed - run: pip install langgraph
# -----------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ Warning: langgraph not installed. Install with: pip install langgraph")

# -----------------------------------------------------------------------------
# NOTE: Google Generative AI is optional - used only for language translation
# and symptom extraction
# -----------------------------------------------------------------------------
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# -----------------------------------------------------------------------------
# NOTE: Database utilities for patient data queries
# Import will fail if database drivers are not installed
# -----------------------------------------------------------------------------
try:
    from db_utils import detect_data_query_intent, handle_data_query
    DB_UTILS_AVAILABLE = True
except ImportError:
    DB_UTILS_AVAILABLE = False
    print("⚠️ Warning: db_utils not available. Patient data queries disabled.")

# -----------------------------------------------------------------------------
# NOTE: Form service for email form integration
# Handles form status checking and response syncing
# -----------------------------------------------------------------------------
try:
    from form_service import check_form_completed, get_form_responses
    FORM_SERVICE_AVAILABLE = True
except ImportError:
    FORM_SERVICE_AVAILABLE = False
    print("⚠️ Warning: form_service not available. Email form integration disabled.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Google API Key - Set this from environment or config file in production
# TODO: Load from environment variable in production
GOOGLE_API_KEY = None  # Set via configure_api() function

# Gemini model instance (initialized lazily)
_gemini_model = None


def configure_api(api_key: str):
    """
    Configure the Google Generative AI API key.
    Call this before using language adaptation or symptom extraction features.
    
    Args:
        api_key: Your Google API key
    """
    global GOOGLE_API_KEY, _gemini_model
    GOOGLE_API_KEY = api_key
    if GENAI_AVAILABLE and api_key:
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel('gemini-pro')


# =============================================================================
# SYSTEM PROMPT (DOCUMENTATION)
# =============================================================================

AGENT_SYSTEM_PROMPT = """
You are Agent 1: a Pharmacovigilance WhatsApp Follow-up Agent.

Your role is to conduct a structured, safety-compliant, visit-based
patient follow-up conversation using WhatsApp.

You do NOT manage databases, reminders, or scheduling.
You ONLY manage conversation flow.

---

### CORE PRINCIPLES (NON-NEGOTIABLE)

1. Each conversation corresponds to exactly ONE visit_id.
2. Patient identity already exists; do not create or modify patient records.
3. Never overwrite past answers.
4. Never infer logic from WhatsApp message history.
5. WhatsApp is UI only; YOU control state.

---

### INPUTS YOU RECEIVE

- visit_id
- patient_id
- is_revisit (true / false)
- last_user_input (text)
- current_state

---

### OUTPUTS YOU MUST PRODUCE

- next_question (MCQ or text)
- next_state
- safety_flag (true / false)
- conversation_complete (true / false)

---

### CONVERSATION FLOW (STRICT ORDER)

1. Language selection
2. Medicine started / continued
3. Adherence (regular / irregular / stopped)
4. Timing & food relation
5. Previous symptom status (if revisit)
6. New symptoms check
7. Severity (if symptoms)
8. Quality & safety check
9. Final closure

---

### SAFETY HARD GATE

If severe symptoms are detected:
- Set safety_flag = true
- Do NOT end conversation
- Escalate to healthcare contact

---

### END CONDITION

Conversation may end ONLY after:
- Quality safety check is complete
- conversation_complete = true
- No unresolved safety flags

---

### FORBIDDEN ACTIONS

You MUST NOT:
- Diagnose any condition
- Give medical advice
- Skip safety checks
- Assume answers from previous messages
- End conversation with unresolved safety flags
"""


# =============================================================================
# STATE DEFINITION
# =============================================================================

class ConversationState(TypedDict, total=False):
    """
    Type definition for Agent 1 conversation state.
    
    All nodes and functions must use only these keys.
    """
    
    # ===== INPUTS (from database trigger) =====
    visit_id: int                    # Current visit ID (CRITICAL)
    patient_id: str                  # Patient identifier
    phone_number: str                # Patient's WhatsApp number
    patient_email: str               # Patient's email for form delivery
    is_revisit: bool                 # True if returning patient
    form_filled: bool                # True if patient filled email form
    
    # ===== FLOW CONTROL =====
    current_state: str               # Current state in flow
    last_user_input: str             # Raw user input
    pending_question: str            # Question waiting for response
    
    # ===== LANGUAGE =====
    preferred_language: str          # Selected language
    
    # ===== MEDICATION DATA =====
    medicine_started: str            # "yes", "no", "tomorrow"
    medicine_continued: str          # "continued", "stopped", "irregular"
    adherence_status: str            # Adherence pattern
    daily_frequency: str             # "once", "twice", "thrice"
    food_relation: str               # "before", "after", "with"
    
    # ===== SYMPTOM DATA =====
    overall_feeling: str             # "better", "same", "worse"
    has_symptoms: bool               # Any symptoms reported
    symptom_description: str         # Raw symptom text
    severity: str                    # "mild", "moderate", "severe"
    time_to_onset: str               # When symptoms started
    body_parts_affected: List[str]   # Affected body parts
    
    # ===== REVISIT-SPECIFIC =====
    previous_symptoms_resolved: str  # "resolved", "ongoing", "worsened"
    new_symptoms_appeared: bool      # Any new symptoms
    
    # ===== SAFETY & OUTPUTS =====
    safety_flag: bool                # BLOCKING - cannot end if True
    safety_reason: str               # Why flag was raised
    conversation_complete: bool      # Can conversation end?
    next_question: str               # Output question text
    next_state: str                  # Output next state
    
    # ===== CALLBACK =====
    callback_scheduled: bool         # Whether callback was scheduled
    callback_method: str             # "phone", "whatsapp", "in_person"


def create_initial_state(
    visit_id: int,
    patient_id: str,
    phone_number: str = "",
    patient_email: str = "",
    is_revisit: bool = False,
    previous_symptoms: str = None
) -> ConversationState:
    """
    Create a clean initial state for a new conversation.
    
    Args:
        visit_id: The visit ID from database (REQUIRED)
        patient_id: Patient identifier (REQUIRED)
        phone_number: Patient's WhatsApp number
        patient_email: Patient's email for form delivery
        is_revisit: Whether this is a returning patient
        previous_symptoms: Symptoms from last visit (for revisits)
    
    Returns:
        Initialized ConversationState
    """
    return ConversationState(
        # Inputs
        visit_id=visit_id,
        patient_id=patient_id,
        phone_number=phone_number,
        patient_email=patient_email,
        is_revisit=is_revisit,
        form_filled=False,
        
        # Flow
        current_state="START",
        last_user_input="",
        pending_question="",
        
        # Language
        preferred_language="English",
        
        # Medication
        medicine_started="",
        medicine_continued="",
        adherence_status="",
        daily_frequency="",
        food_relation="",
        
        # Symptoms
        overall_feeling="",
        has_symptoms=False,
        symptom_description="",
        severity="",
        time_to_onset="",
        body_parts_affected=[],
        
        # Revisit
        previous_symptoms_resolved="" if is_revisit else None,
        new_symptoms_appeared=False,
        
        # Safety & Output
        safety_flag=False,
        safety_reason="",
        conversation_complete=False,
        next_question="",
        next_state="",
        
        # Callback
        callback_scheduled=False,
        callback_method=""
    )


# =============================================================================
# FLOW STATES ENUM
# =============================================================================

class FlowState(Enum):
    """All possible states in the conversation flow."""
    START = "START"
    LANGUAGE_SELECT = "LANGUAGE_SELECT"
    MEDICINE_CHECK = "MEDICINE_CHECK"
    MEDICINE_STARTED = "MEDICINE_STARTED"
    MEDICINE_CONTINUED = "MEDICINE_CONTINUED"
    ADHERENCE = "ADHERENCE"
    TIMING = "TIMING"
    PREVIOUS_SYMPTOMS = "PREVIOUS_SYMPTOMS"  # Revisit only
    FEELING_CHECK = "FEELING_CHECK"
    NEW_SYMPTOMS = "NEW_SYMPTOMS"
    SYMPTOM_DESCRIBE = "SYMPTOM_DESCRIBE"
    ONSET = "ONSET"
    SEVERITY = "SEVERITY"
    BODY_PARTS = "BODY_PARTS"
    SAFETY_CHECK = "SAFETY_CHECK"
    SAFETY_ESCALATION = "SAFETY_ESCALATION"
    CALLBACK_REQUEST = "CALLBACK_REQUEST"
    END_COMPLETE = "END_COMPLETE"
    END_NO_MEDICINE = "END_NO_MEDICINE"
    END_CALLBACK_SCHEDULED = "END_CALLBACK_SCHEDULED"
    SCHEDULE_REMINDER = "SCHEDULE_REMINDER"
    STOPPED_REASON = "STOPPED_REASON"
    IRREGULAR_DETAILS = "IRREGULAR_DETAILS"


# =============================================================================
# QUESTION DEFINITIONS
# =============================================================================

QUESTIONS = {
    # ========== LANGUAGE SELECTION (FIRST QUESTION - MANDATORY) ==========
    "LANG_SELECT": {
        "question_id": "Q1_language",
        "text": """
🌐 Welcome! Please select your preferred language:

1️⃣ English
2️⃣ हिंदी (Hindi)
3️⃣ தமிழ் (Tamil)
4️⃣ తెలుగు (Telugu)
5️⃣ മലയാളം (Malayalam)
        """,
        "options": {"1": "English", "2": "Hindi", "3": "Tamil", "4": "Telugu", "5": "Malayalam"},
        "next_state": "MEDICINE_CHECK"
    },
    
    # ========== NEW PATIENT: MEDICINE STARTED ==========
    "MEDICINE_STARTED": {
        "question_id": "Q2_medicine_started",
        "text": """
💊 Have you started taking the prescribed medicine?

1️⃣ Yes, I have started
2️⃣ No, not yet
3️⃣ I will start tomorrow
        """,
        "options": {"1": "yes", "2": "no", "3": "tomorrow"},
        "next_state_map": {"1": "ADHERENCE", "2": "END_NO_MEDICINE", "3": "SCHEDULE_REMINDER"}
    },
    
    # ========== REVISIT: MEDICINE CONTINUED ==========
    "MEDICINE_CONTINUED": {
        "question_id": "Q2_medicine_continued",
        "text": """
💊 Are you still taking the medicine as prescribed?

1️⃣ Yes, continued regularly
2️⃣ Stopped taking
3️⃣ Taking irregularly
        """,
        "options": {"1": "continued", "2": "stopped", "3": "irregular"},
        "next_state_map": {"1": "ADHERENCE", "2": "STOPPED_REASON", "3": "IRREGULAR_DETAILS"}
    },
    
    # ========== ADHERENCE ==========
    "ADHERENCE": {
        "question_id": "Q3_adherence",
        "text": """
⏰ How often are you taking the medicine daily?

1️⃣ Once a day
2️⃣ Twice a day
3️⃣ Three times a day
4️⃣ As needed
        """,
        "options": {"1": "once", "2": "twice", "3": "thrice", "4": "as_needed"},
        "next_state": "TIMING"
    },
    
    # ========== TIMING & FOOD ==========
    "TIMING": {
        "question_id": "Q4_food_relation",
        "text": """
🍽️ When do you take the medicine in relation to food?

1️⃣ Before food
2️⃣ After food
3️⃣ With food
4️⃣ Empty stomach
        """,
        "options": {"1": "before", "2": "after", "3": "with", "4": "empty"},
        "next_state": "FEELING_CHECK"
    },
    
    # ========== REVISIT: PREVIOUS SYMPTOMS STATUS ==========
    "PREVIOUS_SYMPTOMS": {
        "question_id": "Q5_previous_symptoms",
        "text": """
📋 Last time you mentioned some symptoms. How are they now?

1️⃣ Completely resolved ✅
2️⃣ Still ongoing
3️⃣ Have gotten worse
        """,
        "options": {"1": "resolved", "2": "ongoing", "3": "worsened"},
        "next_state_map": {"1": "NEW_SYMPTOMS", "2": "SEVERITY", "3": "SEVERITY"}
    },
    
    # ========== OVERALL FEELING ==========
    "FEELING_CHECK": {
        "question_id": "Q6_overall_feeling",
        "text": """
💭 Since starting the medicine, how are you feeling overall?

1️⃣ Better 😊
2️⃣ Same 😐
3️⃣ Worse 😔
4️⃣ Much worse 😰
        """,
        "options": {"1": "better", "2": "same", "3": "worse", "4": "much_worse"},
        "next_state_map": {"1": "SAFETY_CHECK", "2": "NEW_SYMPTOMS", "3": "NEW_SYMPTOMS", "4": "URGENT_SYMPTOMS"}
    },
    
    # ========== NEW SYMPTOMS CHECK ==========
    "NEW_SYMPTOMS": {
        "question_id": "Q7_new_symptoms",
        "text": """
🔍 Have you noticed any new symptoms or discomfort?

1️⃣ Yes, I have new symptoms
2️⃣ No new symptoms
        """,
        "options": {"1": "yes", "2": "no"},
        "next_state_map": {"1": "SYMPTOM_DESCRIBE", "2": "SAFETY_CHECK"}
    },
    
    # ========== SYMPTOM DESCRIPTION (Open text) ==========
    "SYMPTOM_DESCRIBE": {
        "question_id": "Q8_symptom_description",
        "text": """
💙 Please describe the symptoms you're experiencing:

(Type your response - be as detailed as possible)
        """,
        "options": None,  # Open text
        "next_state": "ONSET"
    },
    
    # ========== TIME TO ONSET ==========
    "ONSET": {
        "question_id": "Q9_onset",
        "text": """
⏰ When did these symptoms start?

1️⃣ After first dose
2️⃣ Within 1 day
3️⃣ After 2-3 days
4️⃣ After more than 3 days
        """,
        "options": {"1": "first_dose", "2": "within_1_day", "3": "2_3_days", "4": "more_than_3_days"},
        "next_state": "SEVERITY"
    },
    
    # ========== SEVERITY ==========
    "SEVERITY": {
        "question_id": "Q10_severity",
        "text": """
📊 How severe are the symptoms?

1️⃣ Mild (noticeable but manageable)
2️⃣ Moderate (uncomfortable, affecting daily life)
3️⃣ Severe (needs medical attention) ⚠️
        """,
        "options": {"1": "mild", "2": "moderate", "3": "severe"},
        "next_state_map": {"1": "BODY_PARTS", "2": "BODY_PARTS", "3": "SAFETY_ESCALATION"}
    },
    
    # ========== BODY PARTS ==========
    "BODY_PARTS": {
        "question_id": "Q11_body_parts",
        "text": """
🩺 Which part of your body is affected?
(Select all that apply - type numbers separated by commas)

1️⃣ Skin
2️⃣ Stomach/Digestive
3️⃣ Head
4️⃣ Chest
5️⃣ Breathing
6️⃣ Other
        """,
        "options": {"1": "skin", "2": "stomach", "3": "head", "4": "chest", "5": "breathing", "6": "other"},
        "next_state": "SAFETY_CHECK"
    },
    
    # ========== SAFETY CHECK (Final) ==========
    "SAFETY_CHECK": {
        "question_id": "Q12_safety_check",
        "text": """
✅ Final Safety Check

Before we end, please confirm:

1️⃣ I understand I should contact my doctor if symptoms worsen
2️⃣ I have questions for a healthcare professional
        """,
        "options": {"1": "confirmed", "2": "has_questions"},
        "next_state_map": {"1": "END_COMPLETE", "2": "CALLBACK_REQUEST"}
    },
    
    # ========== CALLBACK REQUEST ==========
    "CALLBACK": {
        "question_id": "Q13_callback",
        "text": """
📞 How would you like to be contacted?

1️⃣ Phone call
2️⃣ WhatsApp message
3️⃣ In-person visit
        """,
        "options": {"1": "phone", "2": "whatsapp", "3": "in_person"},
        "next_state": "END_CALLBACK_SCHEDULED"
    }
}


# =============================================================================
# INPUT VALIDATION & EMPATHETIC RESPONSES
# =============================================================================

def validate_input(user_input: str, question_key: str) -> dict:
    """
    Validate if user input is a valid option for the given question.
    
    Args:
        user_input: The user's raw input
        question_key: The question key (e.g., 'LANG_SELECT', 'MEDICINE_STARTED')
        
    Returns:
        Dict with 'valid' bool and 'error_message' if invalid
    """
    question = QUESTIONS.get(question_key)
    if not question:
        return {"valid": True}  # Unknown question, pass through
    
    options = question.get("options")
    if not options:
        return {"valid": True}  # Open text question, any input is valid
    
    # Check if input is a valid option
    clean_input = user_input.strip()
    if clean_input in options:
        return {"valid": True}
    
    # Generate helpful error message
    valid_options = list(options.keys())
    max_option = max(valid_options, key=lambda x: int(x) if x.isdigit() else 0)
    
    error_message = f"""⚠️ Oops! That's not a valid option.

Please reply with a number from 1 to {max_option}.

---
{question.get('text', '')}"""
    
    return {"valid": False, "error_message": error_message}


def get_empathetic_prefix(state: dict, question_key: str) -> str:
    """
    Generate a warm, empathetic prefix message based on context.
    
    Args:
        state: Current conversation state
        question_key: The question being asked
        
    Returns:
        Empathetic prefix string
    """
    prefixes = {
        "MEDICINE_STARTED": "Great! Now let's check on your medication. 💊\n\n",
        "MEDICINE_CONTINUED": "Let's see how your treatment is going. 💊\n\n",
        "ADHERENCE": "Thank you for sharing! 🙏 This helps us understand your routine better.\n\n",
        "TIMING": "Perfect! Just one more question about your routine.\n\n",
        "FEELING_CHECK": "I appreciate you sticking with the questions! 💪 Let's check on how you're feeling.\n\n",
        "NEW_SYMPTOMS": "Thank you for that update. Now, let's make sure everything is okay.\n\n",
        "SYMPTOM_DESCRIBE": "I'm sorry to hear that. 😟 Please share more so we can help better.\n\n",
        "ONSET": "Thank you for sharing. This information is important.\n\n",
        "SEVERITY": "I understand. Let's assess this together.\n\n",
        "BODY_PARTS": "Almost done! This helps us understand better.\n\n",
        "SAFETY_CHECK": "You're doing great! 🌟 Just one final step.\n\n",
        "PREVIOUS_SYMPTOMS": "Welcome back! 👋 Let's check on your previous symptoms.\n\n"
    }
    
    return prefixes.get(question_key, "")


def make_response_empathetic(response: str, user_answer: str, context: str = "") -> str:
    """
    Use Gemini to make responses more empathetic and human.
    Falls back to predefined responses if Gemini unavailable.
    
    Args:
        response: The original response/question
        user_answer: What the user just answered
        context: Additional context
        
    Returns:
        More empathetic version of the response
    """
    global _gemini_model
    
    # Quick empathetic acknowledgments based on common answers
    acknowledgments = {
        "yes": "That's wonderful! 😊 ",
        "no": "I understand. ",
        "1": "Great choice! ",
        "2": "Noted! ",
        "3": "I see. ",
        "better": "That's fantastic news! 🎉 ",
        "same": "I understand. We'll keep monitoring. ",
        "worse": "I'm sorry to hear that. 😟 Let's look into this. ",
        "much_worse": "I'm really sorry you're feeling this way. 💙 Your health is our priority. ",
        "mild": "Thank you for letting us know. ",
        "moderate": "I appreciate you sharing this with us. ",
        "severe": "Thank you for telling us. Your wellbeing is important to us. 💙 ",
        "tomorrow": "No problem! 👍 We'll remind you. ",
        "stopped": "I understand. It's important we know this. ",
        "irregular": "Thank you for being honest. This helps us support you better. "
    }
    
    # Check if we can use Gemini for enhanced empathy
    if _gemini_model and GENAI_AVAILABLE:
        try:
            prompt = f"""You are a caring healthcare assistant. Make this response more warm and empathetic.
Keep it concise (2 sentences max for acknowledgment).
Do NOT change the question itself, only add a brief empathetic acknowledgment before it.
Do NOT give medical advice.

User just answered: {user_answer}
Original response: {response}

Respond with ONLY the improved message, nothing else:"""
            
            result = _gemini_model.generate_content(prompt)
            if result and result.text:
                return result.text.strip()
        except Exception:
            pass  # Fall back to predefined
    
    # Fallback: Use predefined acknowledgments
    ack = acknowledgments.get(user_answer.lower().strip(), "")
    return f"{ack}{response}" if ack else response


# =============================================================================
# RESPONSE STORAGE HOOKS
# =============================================================================
# NOTE: These are stub functions. In production, connect to your database.

def save_response(visit_id: int, question_id: str, answer: str):
    """
    Save a response. NEVER updates — always inserts new row.
    
    TODO: Connect to actual database in production
    
    Args:
        visit_id: The visit ID
        question_id: Question identifier
        answer: User's answer
    """
    # Stub implementation - override with actual database logic
    pass  # TODO: Implement database storage


def save_adverse_event(
    visit_id: int, 
    symptom_description: str, 
    severity: str, 
    time_to_onset: str = None,
    body_parts: list = None, 
    safety_flag: bool = False
):
    """
    Save adverse event data for a visit.
    
    TODO: Connect to actual database in production
    
    Args:
        visit_id: The visit ID
        symptom_description: Raw symptom text
        severity: "mild", "moderate", or "severe"
        time_to_onset: When symptoms started
        body_parts: List of affected body parts
        safety_flag: Whether safety flag was triggered
    """
    # Stub implementation - override with actual database logic
    pass  # TODO: Implement database storage


def schedule_followup(visit_id: int, days_from_now: int, followup_type: str = "reminder"):
    """
    Schedule a follow-up reminder.
    
    TODO: Connect to actual scheduler in production
    
    Args:
        visit_id: The visit ID
        days_from_now: Number of days until follow-up
        followup_type: "reminder" or "escalation"
    """
    # Stub implementation - override with actual scheduler logic
    pass  # TODO: Implement scheduler


# =============================================================================
# NODE FUNCTIONS (FLOW ENGINE)
# =============================================================================

def node_start(state: dict) -> dict:
    """Entry point - route to language selection (FIRST QUESTION - MANDATORY)."""
    state["current_state"] = FlowState.LANGUAGE_SELECT.value
    state["next_question"] = QUESTIONS["LANG_SELECT"]["text"]
    state["next_state"] = FlowState.LANGUAGE_SELECT.value
    return state


def node_language_select(state: dict) -> dict:
    """Process language selection."""
    user_input = state.get("last_user_input", "1")
    options = QUESTIONS["LANG_SELECT"]["options"]
    
    state["preferred_language"] = options.get(user_input, "English")
    
    # Save response
    save_response(state["visit_id"], "Q1_language", state["preferred_language"])
    
    # Route based on revisit status
    if state.get("is_revisit"):
        state["next_state"] = "MEDICINE_CONTINUED"
        state["next_question"] = QUESTIONS["MEDICINE_CONTINUED"]["text"]
    else:
        state["next_state"] = "MEDICINE_STARTED"
        state["next_question"] = QUESTIONS["MEDICINE_STARTED"]["text"]
    
    state["current_state"] = state["next_state"]
    return state


def node_medicine_check(state: dict) -> dict:
    """Process medicine started/continued response."""
    user_input = state.get("last_user_input", "")
    
    if state.get("is_revisit"):
        # Revisit flow
        options = QUESTIONS["MEDICINE_CONTINUED"]["options"]
        state["medicine_continued"] = options.get(user_input, "continued")
        save_response(state["visit_id"], "Q2_medicine_continued", state["medicine_continued"])
        
        if state["medicine_continued"] == "stopped":
            state["next_state"] = "STOPPED_REASON"
            state["next_question"] = "Why did you stop taking the medicine?"
        elif state["medicine_continued"] == "irregular":
            state["next_state"] = "IRREGULAR_DETAILS"
            state["next_question"] = "Can you tell us more about your irregular intake?"
        else:
            state["next_state"] = "ADHERENCE"
            state["next_question"] = QUESTIONS["ADHERENCE"]["text"]
    else:
        # New patient flow
        options = QUESTIONS["MEDICINE_STARTED"]["options"]
        state["medicine_started"] = options.get(user_input, "yes")
        save_response(state["visit_id"], "Q2_medicine_started", state["medicine_started"])
        
        if state["medicine_started"] == "no":
            state["next_state"] = "END_NO_MEDICINE"
            state["next_question"] = """📋 *Noted!*

It's important to start taking your medicine as prescribed by your doctor.

💊 Please start your medication when you can, and we'll follow up with you in a few days.

🙏 Take care and stay healthy!

If you have any concerns about starting your medication, please contact your healthcare provider."""
            state["conversation_complete"] = True
        elif state["medicine_started"] == "tomorrow":
            state["next_state"] = "SCHEDULE_REMINDER"
            schedule_followup(state["visit_id"], 2, "reminder")
            state["next_question"] = """👍 *No problem!*

We understand. We'll remind you in a couple of days to check in.

📅 A follow-up reminder has been scheduled for you.

💊 Please remember to start your medicine as prescribed by your doctor.

🙏 Take care and we'll talk soon!"""
            state["conversation_complete"] = True
        else:
            state["next_state"] = "ADHERENCE"
            state["next_question"] = QUESTIONS["ADHERENCE"]["text"]
    
    state["current_state"] = state["next_state"]
    return state


def node_adherence(state: dict) -> dict:
    """Process adherence response."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["ADHERENCE"]["options"]
    
    state["daily_frequency"] = options.get(user_input, "once")
    save_response(state["visit_id"], "Q3_adherence", state["daily_frequency"])
    
    state["next_state"] = "TIMING"
    state["next_question"] = QUESTIONS["TIMING"]["text"]
    state["current_state"] = state["next_state"]
    return state


def node_timing(state: dict) -> dict:
    """Process timing/food relation response."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["TIMING"]["options"]
    
    state["food_relation"] = options.get(user_input, "after")
    save_response(state["visit_id"], "Q4_food_relation", state["food_relation"])
    
    # Route based on revisit
    if state.get("is_revisit"):
        state["next_state"] = "PREVIOUS_SYMPTOMS"
        state["next_question"] = QUESTIONS["PREVIOUS_SYMPTOMS"]["text"]
    else:
        state["next_state"] = "FEELING_CHECK"
        state["next_question"] = QUESTIONS["FEELING_CHECK"]["text"]
    
    state["current_state"] = state["next_state"]
    return state


def node_previous_symptoms(state: dict) -> dict:
    """Process previous symptoms status (revisit only)."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["PREVIOUS_SYMPTOMS"]["options"]
    
    state["previous_symptoms_resolved"] = options.get(user_input, "resolved")
    save_response(state["visit_id"], "Q5_previous_symptoms", state["previous_symptoms_resolved"])
    
    if state["previous_symptoms_resolved"] == "resolved":
        state["next_state"] = "NEW_SYMPTOMS"
        state["next_question"] = QUESTIONS["NEW_SYMPTOMS"]["text"]
    else:
        state["next_state"] = "SEVERITY"
        state["next_question"] = QUESTIONS["SEVERITY"]["text"]
    
    state["current_state"] = state["next_state"]
    return state


def node_feeling_check(state: dict) -> dict:
    """Process overall feeling response."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["FEELING_CHECK"]["options"]
    
    state["overall_feeling"] = options.get(user_input, "same")
    save_response(state["visit_id"], "Q6_overall_feeling", state["overall_feeling"])
    
    if state["overall_feeling"] == "better":
        state["next_state"] = "SAFETY_CHECK"
        state["next_question"] = QUESTIONS["SAFETY_CHECK"]["text"]
    elif state["overall_feeling"] == "much_worse":
        # SAFETY TRIGGER
        state["next_state"] = "SAFETY_ESCALATION"
        state["safety_flag"] = True
        state["safety_reason"] = "Patient feeling much worse"
    else:
        state["next_state"] = "NEW_SYMPTOMS"
        state["next_question"] = QUESTIONS["NEW_SYMPTOMS"]["text"]
    
    state["current_state"] = state["next_state"]
    return state


def node_new_symptoms(state: dict) -> dict:
    """Process new symptoms check."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["NEW_SYMPTOMS"]["options"]
    
    has_new = options.get(user_input, "no") == "yes"
    state["new_symptoms_appeared"] = has_new
    save_response(state["visit_id"], "Q7_new_symptoms", "yes" if has_new else "no")
    
    if has_new:
        state["next_state"] = "SYMPTOM_DESCRIBE"
        state["next_question"] = QUESTIONS["SYMPTOM_DESCRIBE"]["text"]
    else:
        state["next_state"] = "SAFETY_CHECK"
        state["next_question"] = QUESTIONS["SAFETY_CHECK"]["text"]
    
    state["current_state"] = state["next_state"]
    return state


def node_symptom_describe(state: dict) -> dict:
    """Process symptom description (open text)."""
    state["symptom_description"] = state.get("last_user_input", "")
    save_response(state["visit_id"], "Q8_symptom_description", state["symptom_description"])
    
    state["has_symptoms"] = True
    state["next_state"] = "ONSET"
    state["next_question"] = QUESTIONS["ONSET"]["text"]
    state["current_state"] = state["next_state"]
    return state


def node_onset(state: dict) -> dict:
    """Process time to onset."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["ONSET"]["options"]
    
    state["time_to_onset"] = options.get(user_input, "unknown")
    save_response(state["visit_id"], "Q9_onset", state["time_to_onset"])
    
    state["next_state"] = "SEVERITY"
    state["next_question"] = QUESTIONS["SEVERITY"]["text"]
    state["current_state"] = state["next_state"]
    return state


def node_severity(state: dict) -> dict:
    """Process severity - CRITICAL for safety routing."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["SEVERITY"]["options"]
    
    state["severity"] = options.get(user_input, "mild")
    save_response(state["visit_id"], "Q10_severity", state["severity"])
    
    # SAFETY GATE: Severe symptoms trigger escalation
    if state["severity"] == "severe":
        state["safety_flag"] = True
        state["safety_reason"] = "Severe symptoms reported"
        state["next_state"] = "SAFETY_ESCALATION"
    else:
        state["next_state"] = "BODY_PARTS"
        state["next_question"] = QUESTIONS["BODY_PARTS"]["text"]
    
    state["current_state"] = state["next_state"]
    return state


def node_body_parts(state: dict) -> dict:
    """Process body parts affected."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["BODY_PARTS"]["options"]
    
    # Parse multiple selections (e.g., "1,3,5")
    parts = []
    for char in user_input.replace(",", " ").split():
        if char in options:
            parts.append(options[char])
    
    state["body_parts_affected"] = parts
    save_response(state["visit_id"], "Q11_body_parts", ",".join(parts))
    
    # Save adverse event
    if state.get("has_symptoms"):
        save_adverse_event(
            visit_id=state["visit_id"],
            symptom_description=state.get("symptom_description", ""),
            severity=state.get("severity", "mild"),
            time_to_onset=state.get("time_to_onset"),
            body_parts=parts,
            safety_flag=state.get("safety_flag", False)
        )
    
    state["next_state"] = "SAFETY_CHECK"
    state["next_question"] = QUESTIONS["SAFETY_CHECK"]["text"]
    state["current_state"] = state["next_state"]
    return state


def node_safety_check(state: dict) -> dict:
    """Final safety check - validates conversation can end."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["SAFETY_CHECK"]["options"]
    
    response = options.get(user_input, "confirmed")
    save_response(state["visit_id"], "Q12_safety_check", response)
    
    if response == "has_questions":
        state["next_state"] = "CALLBACK_REQUEST"
        state["next_question"] = QUESTIONS["CALLBACK"]["text"]
    else:
        state["next_state"] = "END_COMPLETE"
        state["next_question"] = """✅ *Thank you for completing this follow-up!*

🌟 Your responses have been recorded and will help your healthcare team monitor your progress.

💊 *Remember:*
• Continue taking your medicine as prescribed
• Contact your doctor if you experience any concerning symptoms
• We're here to support you on your health journey

🙏 Take care and stay healthy! We'll check in with you again soon."""
        state["conversation_complete"] = True
    
    state["current_state"] = state["next_state"]
    return state


def node_safety_escalation(state: dict) -> dict:
    """BLOCKING: Handle safety escalation - CANNOT end with unresolved safety."""
    state["safety_flag"] = True
    
    escalation_message = """
⚠️ IMPORTANT SAFETY NOTICE ⚠️

Based on your responses, we recommend you contact your healthcare provider as soon as possible.

📞 Please call your doctor or visit the nearest healthcare facility.

Would you like us to arrange a callback?

1️⃣ Yes, please arrange a callback
2️⃣ I will contact my doctor myself
    """
    
    state["next_question"] = escalation_message
    state["next_state"] = "CALLBACK_REQUEST"
    state["current_state"] = state["next_state"]
    return state


def node_callback_request(state: dict) -> dict:
    """Process callback request."""
    user_input = state.get("last_user_input", "")
    options = QUESTIONS["CALLBACK"]["options"]
    
    callback_method = options.get(user_input, "phone")
    state["callback_method"] = callback_method
    save_response(state["visit_id"], "Q13_callback", callback_method)
    
    # Schedule callback
    schedule_followup(state["visit_id"], 0, "escalation")
    state["callback_scheduled"] = True
    
    state["next_state"] = "END_CALLBACK_SCHEDULED"
    state["conversation_complete"] = True
    state["current_state"] = state["next_state"]
    return state


def node_end(state: dict) -> dict:
    """End conversation - validate and close."""
    
    # BLOCKING CHECK: Cannot end with unresolved safety flag
    if state.get("safety_flag") and not state.get("callback_scheduled"):
        state["conversation_complete"] = False
        state["next_state"] = "SAFETY_ESCALATION"
        return state
    
    state["conversation_complete"] = True
    
    # Closing message
    end_messages = {
        "END_COMPLETE": "✅ Thank you for completing this follow-up! Take care and stay healthy. 🌟",
        "END_NO_MEDICINE": "📋 Noted. Please start your medicine as prescribed and we'll follow up soon.",
        "END_CALLBACK_SCHEDULED": "📞 A healthcare professional will contact you shortly. Stay safe!",
        "SCHEDULE_REMINDER": "📅 We'll check in with you in a couple of days. Take care!"
    }
    
    state["next_question"] = end_messages.get(state["current_state"], "Thank you!")
    return state


# =============================================================================
# STATE ROUTING
# =============================================================================

def route_next(state: dict) -> str:
    """Determine next node based on state."""
    next_state = state.get("next_state", "")
    
    # Map state names to node names
    state_to_node = {
        "LANGUAGE_SELECT": "LANGUAGE_SELECT",
        "MEDICINE_STARTED": "MEDICINE_CHECK",
        "MEDICINE_CONTINUED": "MEDICINE_CHECK",
        "ADHERENCE": "ADHERENCE",
        "TIMING": "TIMING",
        "PREVIOUS_SYMPTOMS": "PREVIOUS_SYMPTOMS",
        "FEELING_CHECK": "FEELING_CHECK",
        "NEW_SYMPTOMS": "NEW_SYMPTOMS",
        "SYMPTOM_DESCRIBE": "SYMPTOM_DESCRIBE",
        "ONSET": "ONSET",
        "SEVERITY": "SEVERITY",
        "BODY_PARTS": "BODY_PARTS",
        "SAFETY_CHECK": "SAFETY_CHECK",
        "SAFETY_ESCALATION": "SAFETY_ESCALATION",
        "CALLBACK_REQUEST": "CALLBACK_REQUEST",
        "END_COMPLETE": "END",
        "END_NO_MEDICINE": "END",
        "END_CALLBACK_SCHEDULED": "END",
        "SCHEDULE_REMINDER": "END",
        "STOPPED_REASON": "END",  # TODO: Add detailed handling
        "IRREGULAR_DETAILS": "END",  # TODO: Add detailed handling
    }
    
    return state_to_node.get(next_state, "END")


# =============================================================================
# LANGUAGE ADAPTATION (JUSTIFIED LLM USE)
# =============================================================================

def adapt_language(text: str, target_language: str) -> str:
    """
    Adapt question text to patient's preferred language using Google Translate.
    
    This function translates questions to the user's preferred language.
    Uses googletrans library which is free and doesn't require an API key.
    
    Args:
        text: Question text in English
        target_language: Target language name (Hindi, Tamil, Telugu, Malayalam)
    
    Returns:
        Translated text, or original if translation fails
    """
    if target_language == "English":
        return text
    
    # Language code mapping
    LANGUAGE_CODES = {
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Malayalam": "ml",
        "Kannada": "kn",
        "Bengali": "bn",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Punjabi": "pa"
    }
    
    lang_code = LANGUAGE_CODES.get(target_language)
    if not lang_code:
        print(f"⚠️ Unknown language: {target_language}, using English")
        return text
    
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, dest=lang_code)
        return result.text
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return text  # Fallback to English


def extract_symptoms_from_text(free_text: str) -> dict:
    """
    Extract structured symptoms from free-text description.
    
    This is a JUSTIFIED LLM use case:
    - Input: Patient's free-text symptom description
    - Output: Structured symptom data
    - Used for documentation, NOT for flow decisions
    
    Args:
        free_text: Patient's symptom description
    
    Returns:
        Dict with symptoms, body_parts, severity_hint
    """
    if not GENAI_AVAILABLE or not _gemini_model:
        return {
            "symptoms": [free_text],
            "body_parts": [],
            "severity_hint": "unknown",
            "_llm_used": False,
            "_reason": "LLM not available"
        }
    
    prompt = f"""
Extract symptoms from this patient description.
Return JSON with: symptoms (list), body_parts (list), severity_hint (string)
Do NOT diagnose. Do NOT give medical advice.

Patient says: "{free_text}"

JSON response:
"""
    
    try:
        response = _gemini_model.generate_content(prompt)
        # Parse JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")
    
    # Fallback: rule-based extraction
    return {
        "symptoms": [free_text],
        "body_parts": [],
        "severity_hint": "unknown",
        "_llm_used": False,
        "_reason": "Fallback to raw text"
    }


# =============================================================================
# FOLLOW-UP REMINDER HOOKS
# =============================================================================

def get_followup_reminder_message(patient_name: str, reminder_type: str = "reminder") -> str:
    """
    Get the follow-up reminder message text.
    
    This function provides the message templates. The actual scheduling
    should be handled by an external scheduler.
    
    Args:
        patient_name: Patient's name
        reminder_type: "reminder" or "escalation"
    
    Returns:
        Formatted reminder message
    """
    if reminder_type == "reminder":
        return f"""
👋 Hi {patient_name}!

We noticed you haven't completed your follow-up.
Your health is important to us!

Please reply with '1' to continue where you left off.
        """
    else:  # escalation
        return f"""
⚠️ Important Follow-up Required

Hi {patient_name},

We've been trying to reach you for your medication follow-up.
Please respond or contact your healthcare provider.

Reply '1' to continue.
        """


# =============================================================================
# CONVERSATIONAL AGENT CLASS (MAIN INTERFACE)
# =============================================================================

class ConversationalAgent:
    """
    Pharmacovigilance WhatsApp Follow-up Agent
    
    This is the main interface for the agent that:
    1. Receives triggers from database (new visit)
    2. Manages conversation state
    3. Processes user inputs
    4. Returns next question to send via WhatsApp
    
    NOTE: This agent uses step-by-step processing, not LangGraph's full execution.
    Each call to process_input() advances the conversation by exactly one step,
    which is required for conversational agents that need to wait for user input.
    
    Usage:
        agent = ConversationalAgent()
        
        # Start new conversation
        result = agent.start_conversation(
            visit_id=123, 
            patient_id="P001",
            phone_number="+1234567890",
            is_revisit=False
        )
        
        # Process user response
        result = agent.process_input(visit_id=123, user_input="1")
    """
    
    def __init__(self):
        """Initialize the conversation agent."""
        self._states = {}  # In-memory state store (replace with DB in production)
        
        # Node function map for step-by-step processing
        self._node_map = {
            "START": node_start,
            "LANGUAGE_SELECT": node_language_select,
            "MEDICINE_CHECK": node_medicine_check,
            "MEDICINE_STARTED": node_medicine_check,
            "MEDICINE_CONTINUED": node_medicine_check,
            "ADHERENCE": node_adherence,
            "TIMING": node_timing,
            "PREVIOUS_SYMPTOMS": node_previous_symptoms,
            "FEELING_CHECK": node_feeling_check,
            "NEW_SYMPTOMS": node_new_symptoms,
            "SYMPTOM_DESCRIBE": node_symptom_describe,
            "ONSET": node_onset,
            "SEVERITY": node_severity,
            "BODY_PARTS": node_body_parts,
            "SAFETY_CHECK": node_safety_check,
            "SAFETY_ESCALATION": node_safety_escalation,
            "CALLBACK_REQUEST": node_callback_request,
            "END_COMPLETE": node_end,
            "END_NO_MEDICINE": node_end,
            "END_CALLBACK_SCHEDULED": node_end,
            "SCHEDULE_REMINDER": node_end,
        }
    
    def start_conversation(
        self, 
        visit_id: int, 
        patient_id: str,
        phone_number: str = "",
        is_revisit: bool = False
    ) -> dict:
        """
        Start a new conversation for a visit.
        Called when database triggers bot (new visit created).
        
        Args:
            visit_id: The visit ID from database
            patient_id: Patient identifier
            phone_number: Patient's WhatsApp number
            is_revisit: Whether this is a returning patient
            
        Returns:
            dict with next_question and current_state
        """
        # Initialize state
        state = create_initial_state(
            visit_id=visit_id,
            patient_id=patient_id,
            phone_number=phone_number,
            is_revisit=is_revisit
        )
        
        # Run through START node to get the first question (language selection)
        result = node_start(state)
        
        # Store the state
        self._states[visit_id] = result
        
        return {
            "visit_id": visit_id,
            "next_question": result.get("next_question", ""),
            "current_state": result.get("current_state", ""),
            "conversation_complete": result.get("conversation_complete", False)
        }
    
    def process_input(self, visit_id: int, user_input: str) -> dict:
        """
        Process user input and return next question.
        Called when user sends a WhatsApp message.
        
        This advances the conversation by exactly ONE step.
        
        Args:
            visit_id: The visit ID
            user_input: Raw text from user
            
        Returns:
            dict with next_question, current_state, safety_flag, conversation_complete
        """
        # Get current state
        state = self._states.get(visit_id)
        if not state:
            return {"error": "No active conversation for this visit"}
        
        if state.get("conversation_complete"):
            return {
                "visit_id": visit_id,
                "next_question": "This conversation has already ended. Please contact your healthcare provider for further assistance.",
                "conversation_complete": True,
                "current_state": state.get("current_state", ""),
                "safety_flag": state.get("safety_flag", False)
            }
        
        # =====================================================================
        # CHECK IF PATIENT FILLED THE EMAIL FORM
        # If form was filled, thank them and end conversation
        # =====================================================================
        if FORM_SERVICE_AVAILABLE and not state.get("form_filled"):
            if check_form_completed(visit_id):
                state["form_filled"] = True
                state["conversation_complete"] = True
                self._states[visit_id] = state
                
                # Get preferred language for thank you message
                preferred_language = state.get("preferred_language", "English")
                
                thank_you_messages = {
                    "English": (
                        "✅ *Thank You!*\n\n"
                        "Since you've already filled the form through email, "
                        "I won't be asking you the follow-up questions.\n\n"
                        "📝 Your responses have been recorded.\n\n"
                        "💬 If you have any problems or concerns about your medication, "
                        "feel free to reach out to me here or contact your healthcare provider.\n\n"
                        "🙏 Take care and get well soon!"
                    ),
                    "Hindi": (
                        "✅ *धन्यवाद!*\n\n"
                        "चूंकि आपने पहले ही ईमेल के माध्यम से फॉर्म भर दिया है, "
                        "मैं आपसे फॉलो-अप प्रश्न नहीं पूछूंगा।\n\n"
                        "📝 आपकी प्रतिक्रियाएं रिकॉर्ड कर ली गई हैं।\n\n"
                        "💬 अगर आपको अपनी दवा के बारे में कोई समस्या या चिंता है, "
                        "तो यहां मुझसे संपर्क करें या अपने डॉक्टर से बात करें।\n\n"
                        "🙏 अपना ख्याल रखें!"
                    ),
                    "Tamil": (
                        "✅ *நன்றி!*\n\n"
                        "நீங்கள் ஏற்கனவே மின்னஞ்சல் மூலம் படிவத்தை நிரப்பியுள்ளதால், "
                        "பின்தொடர்தல் கேள்விகளை நான் கேட்க மாட்டேன்.\n\n"
                        "📝 உங்கள் பதில்கள் பதிவு செய்யப்பட்டுள்ளன।\n\n"
                        "💬 உங்கள் மருந்து பற்றி ஏதேனும் சிக்கல் இருந்தால், "
                        "என்னை தொடர்பு கொள்ளுங்கள்.\n\n"
                        "🙏 நல்வாழ்த்துக்கள்!"
                    )
                }
                
                thank_you = thank_you_messages.get(preferred_language, thank_you_messages["English"])
                
                return {
                    "visit_id": visit_id,
                    "next_question": thank_you,
                    "current_state": "END_FORM_FILLED",
                    "safety_flag": False,
                    "conversation_complete": True,
                    "form_filled": True
                }
        
        # =====================================================================
        # CHECK FOR DATA QUERY REQUEST
        # If patient asks for their data, handle it before normal flow
        # =====================================================================
        if DB_UTILS_AVAILABLE:
            query_intent = detect_data_query_intent(user_input)
            if query_intent.get('is_data_query'):
                phone_number = state.get("phone_number", "")
                query_type = query_intent.get('query_type', 'full_data')
                specific_visit_id = query_intent.get('visit_id')
                
                # Get and format patient data
                data_response = handle_data_query(
                    phone_number=phone_number,
                    query_type=query_type,
                    visit_id=specific_visit_id
                )
                
                # Return data without advancing conversation state
                # Include reminder about the pending question
                pending_question = state.get("pending_question", state.get("next_question", ""))
                if pending_question:
                    data_response += f"\n\n---\n📝 *Pending Question:*\n{pending_question}"
                
                return {
                    "visit_id": visit_id,
                    "next_question": data_response,
                    "current_state": state.get("current_state", ""),
                    "safety_flag": state.get("safety_flag", False),
                    "conversation_complete": False,
                    "is_data_query_response": True  # Flag to indicate this was a data query
                }
        
        # =====================================================================
        # VALIDATE USER INPUT BEFORE PROCESSING
        # =====================================================================
        current = state.get("current_state", "START")
        
        # Map current state to question key for validation
        state_to_question_key = {
            "LANGUAGE_SELECT": "LANG_SELECT",
            "MEDICINE_STARTED": "MEDICINE_STARTED",
            "MEDICINE_CONTINUED": "MEDICINE_CONTINUED",
            "ADHERENCE": "ADHERENCE",
            "TIMING": "TIMING",
            "PREVIOUS_SYMPTOMS": "PREVIOUS_SYMPTOMS",
            "FEELING_CHECK": "FEELING_CHECK",
            "NEW_SYMPTOMS": "NEW_SYMPTOMS",
            "SYMPTOM_DESCRIBE": "SYMPTOM_DESCRIBE",  # Open text - no validation
            "ONSET": "ONSET",
            "SEVERITY": "SEVERITY",
            "BODY_PARTS": "BODY_PARTS",
            "SAFETY_CHECK": "SAFETY_CHECK",
            "CALLBACK_REQUEST": "CALLBACK"
        }
        
        question_key = state_to_question_key.get(current)
        
        # Validate input for MCQ questions
        if question_key:
            validation = validate_input(user_input, question_key)
            if not validation.get("valid"):
                # Invalid input - return error and repeat question
                preferred_language = state.get("preferred_language", "English")
                error_msg = validation.get("error_message", "Please enter a valid option.")
                
                # Translate error if needed
                if preferred_language != "English":
                    error_msg = adapt_language(error_msg, preferred_language)
                
                return {
                    "visit_id": visit_id,
                    "next_question": error_msg,
                    "current_state": current,
                    "safety_flag": state.get("safety_flag", False),
                    "conversation_complete": False,
                    "validation_error": True
                }
        
        # Update state with user input
        state["last_user_input"] = user_input.strip()
        
        # Get the appropriate node function for the current state
        node_func = self._node_map.get(current, node_end)
        
        # Process ONE step
        result = node_func(state)
        
        # Store updated state
        self._states[visit_id] = result
        
        # Get the next question and adapt to user's language
        next_question = result.get("next_question", "")
        preferred_language = result.get("preferred_language", "English")
        
        # =====================================================================
        # ADD EMPATHETIC PREFIX (makes responses feel more human)
        # =====================================================================
        next_state = result.get("next_state", result.get("current_state", ""))
        empathetic_prefix = get_empathetic_prefix(state, next_state)
        
        # Add empathetic prefix for a warmer response
        if next_question and not result.get("conversation_complete"):
            if empathetic_prefix and not next_question.startswith(empathetic_prefix):
                next_question = empathetic_prefix + next_question
        
        # Translate the question if not English
        if next_question and preferred_language != "English":
            next_question = adapt_language(next_question, preferred_language)
        
        return {
            "visit_id": visit_id,
            "next_question": next_question,
            "current_state": result.get("current_state", ""),
            "safety_flag": result.get("safety_flag", False),
            "conversation_complete": result.get("conversation_complete", False)
        }
    
    def get_status(self, visit_id: int) -> dict:
        """Get current conversation status for a visit."""
        state = self._states.get(visit_id)
        if not state:
            return {"error": "No conversation found"}
        
        return {
            "visit_id": visit_id,
            "current_state": state.get("current_state", ""),
            "conversation_complete": state.get("conversation_complete", False),
            "safety_flag": state.get("safety_flag", False),
            "preferred_language": state.get("preferred_language", "English")
        }
    
    def get_state(self, visit_id: int) -> dict:
        """Get full conversation state for a visit."""
        return self._states.get(visit_id, {})


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_mcq_input(user_input: str, valid_options: dict) -> tuple:
    """
    Validate MCQ input from user.
    
    Args:
        user_input: Raw user input
        valid_options: Dict of valid option numbers to values
    
    Returns:
        Tuple of (is_valid, normalized_input)
    """
    normalized = user_input.strip()
    if normalized in valid_options:
        return True, normalized
    return False, None


def get_question_by_state(state_name: str) -> dict:
    """
    Get question definition by state name.
    
    Args:
        state_name: Current flow state name
    
    Returns:
        Question definition dict or empty dict
    """
    state_to_question = {
        "LANGUAGE_SELECT": "LANG_SELECT",
        "MEDICINE_STARTED": "MEDICINE_STARTED",
        "MEDICINE_CONTINUED": "MEDICINE_CONTINUED",
        "ADHERENCE": "ADHERENCE",
        "TIMING": "TIMING",
        "PREVIOUS_SYMPTOMS": "PREVIOUS_SYMPTOMS",
        "FEELING_CHECK": "FEELING_CHECK",
        "NEW_SYMPTOMS": "NEW_SYMPTOMS",
        "SYMPTOM_DESCRIBE": "SYMPTOM_DESCRIBE",
        "ONSET": "ONSET",
        "SEVERITY": "SEVERITY",
        "BODY_PARTS": "BODY_PARTS",
        "SAFETY_CHECK": "SAFETY_CHECK",
        "CALLBACK_REQUEST": "CALLBACK",
    }
    
    question_key = state_to_question.get(state_name)
    return QUESTIONS.get(question_key, {})


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Create a default agent instance
_default_agent = None


def get_default_agent() -> ConversationalAgent:
    """Get or create the default agent instance."""
    global _default_agent
    if _default_agent is None:
        _default_agent = ConversationalAgent()
    return _default_agent


# =============================================================================
# EXAMPLE USAGE (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📋 ConversationalAgent Module Loaded")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from ConversationalAgent import ConversationalAgent")
    print()
    print("  agent = ConversationalAgent()")
    print("  result = agent.start_conversation(visit_id=1, patient_id='P001')")
    print("  print(result['next_question'])")
    print()
    print("  result = agent.process_input(visit_id=1, user_input='1')")
    print("  print(result['next_question'])")
    print()
    
    # Quick test
    agent = ConversationalAgent()
    
    print("=" * 60)
    print("🧪 Running Quick Test")
    print("=" * 60)
    
    # Start conversation
    result = agent.start_conversation(
        visit_id=1, 
        patient_id="TEST_P001",
        phone_number="+1234567890",
        is_revisit=False
    )
    print(f"\n📤 [Bot]: {result['next_question'][:100]}...")
    print(f"   State: {result['current_state']}")
    
    # Simulate language selection
    result = agent.process_input(visit_id=1, user_input="1")
    print(f"\n📥 [User]: 1 (English)")
    print(f"📤 [Bot]: {result['next_question'][:100]}...")
    print(f"   State: {result['current_state']}")
    
    print("\n✅ Quick test completed!")
    print("=" * 60)
