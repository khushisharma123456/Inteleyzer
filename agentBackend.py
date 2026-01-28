"""
🔗 AGENT BACKEND - Twilio WhatsApp Integration for Pharmacovigilance Agent

This module connects the ConversationalAgent to Twilio for WhatsApp messaging.

PIPELINE FLOW:
    Database Trigger → POST /start-conversation → Agent starts → Twilio sends message
    Patient replies  → Twilio webhook → POST /whatsapp → Agent processes → Next question

Author: Corrected for production-ready Twilio integration
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv
import os
import logging
import json

# =============================================================================
# IMPORT THE CONVERSATIONAL AGENT
# =============================================================================
# This is the core agent that manages conversation flow and state
from ConversationalAgent import ConversationalAgent, get_followup_reminder_message

# =============================================================================
# IMPORT EMAIL AND FORM SERVICES
# =============================================================================
try:
    from email_service import send_form_email, send_clarification_email
    EMAIL_SERVICE_AVAILABLE = True
except ImportError:
    EMAIL_SERVICE_AVAILABLE = False
    print("⚠️ Warning: email_service not available")

try:
    from form_service import (
        generate_form_url, validate_form_token, process_form_submission,
        get_questions_for_language, LANGUAGE_NAMES
    )
    FORM_SERVICE_AVAILABLE = True
except ImportError:
    FORM_SERVICE_AVAILABLE = False
    print("⚠️ Warning: form_service not available")

# Import for serving HTML templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Pharmacovigilance WhatsApp Agent",
    description="Twilio-powered WhatsApp follow-up system for medication monitoring",
    version="1.0.0"
)

# Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_FROM = os.getenv('TWILIO_WHATSAPP_FROM')

# Initialize Twilio client (only if credentials are available)
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logger.info("✅ Twilio client initialized")
else:
    logger.warning("⚠️ Twilio credentials not found - running in test mode")

# =============================================================================
# INITIALIZE THE CONVERSATIONAL AGENT
# =============================================================================
# Single agent instance that manages all conversations
agent = ConversationalAgent()

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================
# Maps phone numbers to their active visit_id
# Format: {"+1234567890": visit_id}
# 
# NOTE: In production, this should be stored in a database
PHONE_TO_VISIT = {}

# Auto-incrementing visit ID (in production, this comes from database)
_next_visit_id = 1


def get_next_visit_id() -> int:
    """Generate the next visit ID. Replace with database logic in production."""
    global _next_visit_id
    visit_id = _next_visit_id
    _next_visit_id += 1
    return visit_id


# =============================================================================
# TWILIO MESSAGE SENDING FUNCTIONS
# =============================================================================

def send_whatsapp_message(to_phone: str, body: str) -> str:
    """
    Send a WhatsApp message via Twilio.
    
    Args:
        to_phone: Phone number with country code (e.g., '+918595687463')
        body: Message text to send
        
    Returns:
        Message SID from Twilio, or None if in test mode
    """
    if not twilio_client:
        logger.info(f"[TEST MODE] Would send to {to_phone}: {body[:50]}...")
        return "TEST_MODE_SID"
    
    try:
        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            body=body,
            to=f'whatsapp:{to_phone}'
        )
        logger.info(f"✅ Sent message to {to_phone}: SID={message.sid}")
        return message.sid
    except Exception as e:
        logger.error(f"❌ Failed to send message: {e}")
        raise


def send_whatsapp_interactive_list(to_phone: str, header: str, body: str, button_text: str, sections: list) -> str:
    """
    Send a WhatsApp interactive list message via Twilio.
    
    This creates a message with a button that opens a list picker (like radio buttons).
    Users tap the button to see options and select one.
    
    Args:
        to_phone: Phone number with country code (e.g., '+918595687463')
        header: Header text for the message
        body: Body text explaining the selection
        button_text: Text on the button that opens the list (max 20 chars)
        sections: List of sections, each containing items:
                  [{"title": "Languages", "items": [{"id": "1", "title": "English"}, ...]}]
        
    Returns:
        Message SID from Twilio, or None if in test mode
    """
    if not twilio_client:
        logger.info(f"[TEST MODE] Would send interactive list to {to_phone}")
        return "TEST_MODE_SID"
    
    try:
        # Build the interactive list payload
        interactive_payload = {
            "type": "list",
            "header": {
                "type": "text",
                "text": header
            },
            "body": {
                "text": body
            },
            "action": {
                "button": button_text,
                "sections": sections
            }
        }
        
        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=f'whatsapp:{to_phone}',
            content_sid=None,  # Not using a template
            persistent_action=[f'interactive:{json.dumps(interactive_payload)}']
        )
        logger.info(f"✅ Sent interactive list to {to_phone}: SID={message.sid}")
        return message.sid
    except Exception as e:
        logger.error(f"❌ Failed to send interactive list: {e}")
        # Fallback to regular message
        logger.info("↩️ Falling back to regular message")
        return send_whatsapp_message(to_phone, f"{header}\n\n{body}")


def send_language_selection(to_phone: str) -> str:
    """
    Send the language selection message with numbered options.
    
    Note: Twilio Sandbox doesn't support interactive list messages,
    so we send plain text with numbered options instead.
    
    Args:
        to_phone: Phone number with country code
        
    Returns:
        Message SID from Twilio
    """
    # Send numbered options as plain text (works with Twilio Sandbox)
    language_message = """🌐 Welcome! Please select your preferred language:

1️⃣ English
2️⃣ हिंदी (Hindi)
3️⃣ தமிழ் (Tamil)
4️⃣ తెలుగు (Telugu)
5️⃣ മലയാളം (Malayalam)

Reply with a number (1-5) to continue."""
    
    return send_whatsapp_message(to_phone, language_message)


def send_whatsapp_reminder(to_phone: str, date: str, time: str) -> str:
    """
    Send a WhatsApp appointment reminder using Twilio template.
    
    Args:
        to_phone: Phone number with country code
        date: Appointment date
        time: Appointment time
        
    Returns:
        Message SID from Twilio
    """
    if not twilio_client:
        logger.info(f"[TEST MODE] Would send reminder to {to_phone}")
        return "TEST_MODE_SID"
    
    message = twilio_client.messages.create(
        from_=TWILIO_WHATSAPP_FROM,
        content_sid='HXb5b62575e6e4ff6129ad7c8efe1f983e',
        content_variables=f'{{"1":"{date}","2":"{time}"}}',
        to=f'whatsapp:{to_phone}'
    )
    return message.sid


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/start-conversation")
async def start_conversation(request: Request):
    """
    Start a new conversation for a patient.
    
    Called when a doctor adds/updates a patient in the database.
    This is the TRIGGER that starts the WhatsApp follow-up flow.
    
    Request body:
    {
        "patient_id": "P001",
        "phone_number": "+918595687463",
        "is_revisit": false,
        "visit_id": 123  // Optional - will auto-generate if not provided
    }
    
    Response:
    {
        "success": true,
        "visit_id": 123,
        "next_question": "🌐 Welcome! Please select your preferred language...",
        "message_sent": true
    }
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    # Extract required fields
    patient_id = data.get("patient_id")
    phone_number = data.get("phone_number")
    patient_email = data.get("email", "")  # Optional email for form delivery
    patient_name = data.get("patient_name", "Patient")  # For email personalization
    is_revisit = data.get("is_revisit", False)
    preferred_language = data.get("language", "en")  # Preferred language code
    
    # Validate required fields
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")
    if not phone_number:
        raise HTTPException(status_code=400, detail="phone_number is required")
    
    # Get visit_id (from database or auto-generate)
    visit_id = data.get("visit_id") or get_next_visit_id()
    
    # Start conversation with the agent
    result = agent.start_conversation(
        visit_id=visit_id,
        patient_id=patient_id,
        phone_number=phone_number,
        is_revisit=is_revisit
    )
    
    # Map phone number to visit_id for incoming message routing
    PHONE_TO_VISIT[phone_number] = visit_id
    
    logger.info(f"📱 Started conversation: visit_id={visit_id}, patient={patient_id}, phone={phone_number}")
    
    # ==========================================================================
    # SEND EMAIL WITH FORM (if email provided)
    # ==========================================================================
    email_sent = False
    form_url = None
    
    if patient_email and EMAIL_SERVICE_AVAILABLE and FORM_SERVICE_AVAILABLE:
        try:
            # Generate unique form URL
            form_url = generate_form_url(visit_id, patient_id, preferred_language)
            
            # Send email with form link
            email_sent = send_form_email(
                to_email=patient_email,
                patient_name=patient_name,
                visit_id=visit_id,
                patient_id=patient_id,
                language=preferred_language
            )
            
            if email_sent:
                logger.info(f"📧 Form email sent to {patient_email}")
            else:
                logger.warning(f"⚠️ Form email not sent to {patient_email}")
                
        except Exception as e:
            logger.error(f"Failed to send form email: {e}")
    
    # ==========================================================================
    # SEND WHATSAPP MESSAGE
    # ==========================================================================
    # Send the first question (language selection) via WhatsApp
    next_question = result.get("next_question", "")
    message_sent = False
    
    if next_question and "language" in next_question.lower():
        # Use interactive list for language selection
        try:
            send_language_selection(phone_number)
            message_sent = True
        except Exception as e:
            logger.error(f"Failed to send language selection: {e}")
            # Fallback to regular message
            try:
                send_whatsapp_message(phone_number, next_question)
                message_sent = True
            except Exception as e2:
                logger.error(f"Failed to send fallback message: {e2}")
    elif next_question:
        try:
            send_whatsapp_message(phone_number, next_question)
            message_sent = True
        except Exception as e:
            logger.error(f"Failed to send initial message: {e}")
    
    return {
        "success": True,
        "visit_id": visit_id,
        "current_state": result.get("current_state", ""),
        "next_question": next_question,
        "message_sent": message_sent,
        "email_sent": email_sent,
        "form_url": form_url
    }


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Twilio WhatsApp Webhook - receives incoming messages from patients.
    
    This is called by Twilio whenever a patient sends a WhatsApp message.
    
    Flow:
    1. Receive message from Twilio (form data)
    2. Look up the active visit_id for this phone number
    3. Pass the message to the ConversationalAgent
    4. Return the next question as TwiML response
    
    Twilio sends form data with:
    - From: "whatsapp:+1234567890"
    - Body: "1" (user's message)
    """
    # Parse the incoming form data from Twilio
    form = await request.form()
    user_msg = form.get("Body", "").strip()
    from_number_raw = form.get("From", "")
    
    # Extract phone number (remove "whatsapp:" prefix)
    phone_number = from_number_raw.replace("whatsapp:", "")
    
    logger.info(f"📥 Received from {phone_number}: '{user_msg}'")
    
    # Look up the active visit for this phone number
    visit_id = PHONE_TO_VISIT.get(phone_number)
    
    # Prepare TwiML response
    twiml = MessagingResponse()
    
    if not visit_id:
        # No active conversation - send a helpful message
        logger.warning(f"⚠️ No active conversation for {phone_number}")
        twiml.message(
            "Hello! You don't have an active follow-up session. "
            "Please wait for your healthcare provider to initiate contact."
        )
        return PlainTextResponse(content=str(twiml), media_type="application/xml")
    
    # Process the input through the ConversationalAgent
    result = agent.process_input(visit_id=visit_id, user_input=user_msg)
    
    logger.info(f"📤 Agent response: state={result.get('current_state')}, complete={result.get('conversation_complete')}")
    
    # Get the next question to send
    next_question = result.get("next_question", "")
    
    # Check if conversation is complete
    if result.get("conversation_complete"):
        # Clean up the phone-to-visit mapping
        if phone_number in PHONE_TO_VISIT:
            del PHONE_TO_VISIT[phone_number]
        logger.info(f"✅ Conversation complete for visit_id={visit_id}")
    
    # Safety flag handling - log for healthcare team
    if result.get("safety_flag"):
        logger.warning(f"🚨 SAFETY FLAG RAISED for visit_id={visit_id}!")
    
    # Log data query responses
    if result.get("is_data_query_response"):
        logger.info(f"📊 Data query response sent to {phone_number}")
    
    # Send the next question as TwiML response
    if next_question:
        twiml.message(next_question)
    else:
        twiml.message("Thank you for your response.")
    
    return PlainTextResponse(content=str(twiml), media_type="application/xml")


@app.get("/conversation-status/{phone_number}")
async def get_conversation_status(phone_number: str):
    """
    Get the current conversation status for a phone number.
    
    Useful for checking if a patient has an active conversation.
    """
    visit_id = PHONE_TO_VISIT.get(phone_number)
    
    if not visit_id:
        return {"active": False, "phone_number": phone_number}
    
    status = agent.get_status(visit_id)
    return {
        "active": True,
        "phone_number": phone_number,
        "visit_id": visit_id,
        **status
    }


@app.post("/send-reminder")
async def send_reminder(request: Request):
    """
    Send appointment reminder via WhatsApp (uses Twilio template).
    
    Request body:
    {
        "phone": "+918595687463",
        "date": "12/1",
        "time": "3pm"
    }
    """
    data = await request.json()
    phone = data.get("phone")
    date = data.get("date")
    time = data.get("time")
    
    if not all([phone, date, time]):
        raise HTTPException(status_code=400, detail="phone, date, and time are required")
    
    try:
        message_sid = send_whatsapp_reminder(phone, date, time)
        return {"success": True, "message_sid": message_sid}
    except Exception as e:
        logger.error(f"Failed to send reminder: {e}")
        return {"success": False, "error": str(e)}


@app.post("/send-followup-reminder")
async def send_followup_reminder(request: Request):
    """
    Send a follow-up reminder to a patient who hasn't responded.
    
    Request body:
    {
        "phone": "+918595687463",
        "patient_name": "John",
        "reminder_type": "reminder"  // or "escalation"
    }
    """
    data = await request.json()
    phone = data.get("phone")
    patient_name = data.get("patient_name", "Patient")
    reminder_type = data.get("reminder_type", "reminder")
    
    if not phone:
        raise HTTPException(status_code=400, detail="phone is required")
    
    # Get the reminder message from the agent module
    message = get_followup_reminder_message(patient_name, reminder_type)
    
    try:
        message_sid = send_whatsapp_message(phone, message)
        return {"success": True, "message_sid": message_sid}
    except Exception as e:
        logger.error(f"Failed to send follow-up reminder: {e}")
        return {"success": False, "error": str(e)}


@app.post("/send-alert")
async def send_alert(request: Request):
    """
    Send a custom alert message via WhatsApp.
    
    Request body:
    {
        "phone": "+918595687463",
        "message": "Your prescription is ready!"
    }
    """
    data = await request.json()
    phone = data.get("phone")
    message = data.get("message")
    
    if not all([phone, message]):
        raise HTTPException(status_code=400, detail="phone and message are required")
    
    try:
        message_sid = send_whatsapp_message(phone, message)
        return {"success": True, "message_sid": message_sid}
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")
        return {"success": False, "error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "twilio_configured": twilio_client is not None,
        "email_service_available": EMAIL_SERVICE_AVAILABLE,
        "form_service_available": FORM_SERVICE_AVAILABLE,
        "active_conversations": len(PHONE_TO_VISIT)
    }


# =============================================================================
# FORM ENDPOINTS
# =============================================================================

@app.get("/form/{token}", response_class=HTMLResponse)
async def serve_form(token: str, lang: str = "en"):
    """
    Serve the patient follow-up form.
    
    Args:
        token: Unique form token
        lang: Language code (en, hi, ta, te, ml)
    """
    # Validate token
    if FORM_SERVICE_AVAILABLE:
        token_data = validate_form_token(token)
        if not token_data:
            return HTMLResponse(
                content="<h1>Form Expired or Invalid</h1><p>This form link is no longer valid.</p>",
                status_code=404
            )
        
        if token_data.get('filled'):
            return HTMLResponse(
                content="<h1>Form Already Submitted</h1><p>Thank you! Your responses have already been recorded.</p>",
                status_code=200
            )
    
    # Read and serve the HTML form template
    template_path = Path(__file__).parent / "templates" / "patient_form.html"
    
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(
            content="<h1>Form Not Found</h1><p>The form template is missing.</p>",
            status_code=500
        )


@app.post("/api/form/submit")
async def submit_form(request: Request):
    """
    Handle form submission from the web form.
    
    Request body:
    {
        "token": "form_token_here",
        "responses": {
            "language": "en",
            "medicine_started": "yes",
            "adherence": "once",
            ...
        }
    }
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    token = data.get("token")
    responses = data.get("responses", {})
    
    if not token:
        raise HTTPException(status_code=400, detail="Form token is required")
    
    if not FORM_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Form service not available")
    
    # Process the form submission
    result = process_form_submission(token, responses)
    
    if result.get('success'):
        logger.info(f"✅ Form submitted: visit_id={result.get('visit_id')}")
        return {
            "success": True,
            "message": "Thank you! Your responses have been recorded."
        }
    else:
        return {
            "success": False,
            "error": result.get('error', 'Unknown error')
        }


@app.post("/send-clarification-form")
async def send_clarification_form(request: Request):
    """
    Send a clarification form to a patient for missing data.
    
    Request body:
    {
        "visit_id": 123,
        "patient_id": "P001",
        "email": "patient@email.com",
        "patient_name": "John",
        "missing_fields": ["symptom_description", "severity"],
        "language": "en"
    }
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    visit_id = data.get("visit_id")
    patient_id = data.get("patient_id")
    email = data.get("email")
    patient_name = data.get("patient_name", "Patient")
    missing_fields = data.get("missing_fields", [])
    language = data.get("language", "en")
    
    if not all([visit_id, patient_id, email, missing_fields]):
        raise HTTPException(
            status_code=400, 
            detail="visit_id, patient_id, email, and missing_fields are required"
        )
    
    if not EMAIL_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Email service not available")
    
    try:
        # Send clarification email
        email_sent = send_clarification_email(
            to_email=email,
            patient_name=patient_name,
            visit_id=visit_id,
            patient_id=patient_id,
            missing_fields=missing_fields,
            language=language
        )
        
        if email_sent:
            logger.info(f"📧 Clarification email sent to {email}")
            return {"success": True, "message": "Clarification form sent"}
        else:
            return {"success": False, "error": "Failed to send email"}
            
    except Exception as e:
        logger.error(f"Failed to send clarification email: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🏥 Pharmacovigilance WhatsApp Agent Backend")
    print("=" * 60)
    print()
    print("📡 Server starting on http://0.0.0.0:8000")
    print()
    print("Endpoints:")
    print("  POST /start-conversation    - Start new patient follow-up")
    print("  POST /whatsapp              - Twilio webhook for incoming messages")
    print("  GET  /conversation-status   - Check conversation status")
    print("  POST /send-reminder         - Send appointment reminder")
    print("  POST /send-alert            - Send custom alert")
    print("  GET  /form/{token}          - Serve patient form")
    print("  POST /api/form/submit       - Handle form submission")
    print("  POST /send-clarification    - Send clarification form")
    print("  GET  /health                - Health check")
    print()
    print(f"📧 Email Service: {'✅ Available' if EMAIL_SERVICE_AVAILABLE else '❌ Not configured'}")
    print(f"📝 Form Service: {'✅ Available' if FORM_SERVICE_AVAILABLE else '❌ Not configured'}")
    print()
    print("🔧 To expose for Twilio:")
    print("  1. Run: ngrok http 8000")
    print("  2. Copy the ngrok URL")
    print("  3. Set Twilio webhook to: https://<ngrok-url>/whatsapp")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
