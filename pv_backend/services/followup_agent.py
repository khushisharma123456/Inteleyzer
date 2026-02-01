"""
Follow-up Agent Service
=======================
Sends follow-up questionnaire emails and WhatsApp messages to patients 
and processes their responses to update case data and improve case scoring.

Setup Gmail:
1. Enable 2-Factor Authentication on your Google Account
2. Go to: Google Account → Security → 2-Step Verification → App passwords
3. Generate an "App password" for Mail
4. Set environment variables:
   - GMAIL_ADDRESS=your-email@gmail.com
   - GMAIL_APP_PASSWORD=your-16-char-app-password

Setup Twilio WhatsApp:
1. Create a Twilio account at https://www.twilio.com
2. Set up WhatsApp Sandbox or approved WhatsApp Business API
3. Set environment variables:
   - TWILIO_ACCOUNT_SID=your-account-sid
   - TWILIO_AUTH_TOKEN=your-auth-token
   - TWILIO_WHATSAPP_FROM=whatsapp:+14155238886  (your Twilio WhatsApp number)
   
Or update the config in this file directly (not recommended for production).
"""

import smtplib
import os
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file (override ensures fresh values)
load_dotenv(override=True)

# Gmail Configuration - Set these via environment variables in .env file
GMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': os.environ.get('GMAIL_ADDRESS'),
    'password': os.environ.get('GMAIL_APP_PASSWORD'),
}

# Twilio WhatsApp Configuration - Set these via environment variables in .env file
TWILIO_CONFIG = {
    'account_sid': os.environ.get('TWILIO_ACCOUNT_SID'),
    'auth_token': os.environ.get('TWILIO_AUTH_TOKEN'),
    'whatsapp_from': os.environ.get('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886'),
}

# Base URL for the application (set in .env file)
# Using ngrok for public access - update APP_BASE_URL in .env when ngrok restarts
APP_BASE_URL = os.environ.get('APP_BASE_URL', 'http://localhost:5000')


class FollowupAgent:
    """
    Agent responsible for sending follow-up questionnaires to patients
    via email and WhatsApp, and processing their responses to update case data.
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.email_config = GMAIL_CONFIG
        self.twilio_config = TWILIO_CONFIG
        self.twilio_client = None
        
        # Initialize Twilio client if credentials are available
        if self.is_whatsapp_configured():
            try:
                from twilio.rest import Client
                self.twilio_client = Client(
                    self.twilio_config['account_sid'],
                    self.twilio_config['auth_token']
                )
            except ImportError:
                print("⚠️ Twilio library not installed. Run: pip install twilio")
            except Exception as e:
                print(f"⚠️ Failed to initialize Twilio client: {e}")
        
    def configure_email(self, email: str, password: str):
        """Configure email credentials programmatically"""
        self.email_config['email'] = email
        self.email_config['password'] = password
    
    def configure_whatsapp(self, account_sid: str, auth_token: str, whatsapp_from: str):
        """Configure Twilio WhatsApp credentials programmatically"""
        self.twilio_config['account_sid'] = account_sid
        self.twilio_config['auth_token'] = auth_token
        self.twilio_config['whatsapp_from'] = whatsapp_from
        
        # Re-initialize client with new credentials
        try:
            from twilio.rest import Client
            self.twilio_client = Client(account_sid, auth_token)
        except Exception as e:
            print(f"⚠️ Failed to initialize Twilio client: {e}")
        
    def is_email_configured(self) -> bool:
        """Check if email credentials are properly configured"""
        return bool(self.email_config['email'] and self.email_config['password'])
    
    def is_whatsapp_configured(self) -> bool:
        """Check if WhatsApp/Twilio credentials are properly configured"""
        return bool(
            self.twilio_config.get('account_sid') and 
            self.twilio_config.get('auth_token') and
            self.twilio_config.get('whatsapp_from')
        )
    
    def generate_followup_token(self, patient_id: str) -> str:
        """Generate a secure token for the follow-up form link"""
        return secrets.token_urlsafe(32)
    
    def get_followup_questions(self, patient) -> list:
        """
        Generate follow-up questions based on patient's current data.
        Questions target missing or incomplete information to improve case scoring.
        """
        questions = []
        
        # Always ask about current status
        questions.append({
            'id': 'current_status',
            'type': 'select',
            'label': 'How are you feeling now regarding the reported side effect?',
            'options': ['Fully Recovered', 'Improving', 'Same as before', 'Worsening', 'New symptoms appeared'],
            'required': True
        })
        
        # Temporal clarity questions
        if not patient.symptom_onset_date:
            questions.append({
                'id': 'symptom_onset_date',
                'type': 'date',
                'label': 'When did the symptoms first appear?',
                'required': False
            })
            
        if not patient.symptom_resolution_date:
            questions.append({
                'id': 'symptom_resolution_date', 
                'type': 'date',
                'label': 'If symptoms have resolved, when did they stop?',
                'required': False
            })
        
        # Medical confirmation
        if not patient.doctor_confirmed:
            questions.append({
                'id': 'doctor_visit',
                'type': 'select',
                'label': 'Have you consulted a doctor about these symptoms?',
                'options': ['Yes', 'No', 'Planning to'],
                'required': False
            })
            
        # Severity update
        questions.append({
            'id': 'current_severity',
            'type': 'select',
            'label': 'How would you rate the severity of symptoms now?',
            'options': ['None', 'Mild', 'Moderate', 'Severe'],
            'required': True
        })
        
        # Additional symptoms
        questions.append({
            'id': 'additional_symptoms',
            'type': 'textarea',
            'label': 'Have you experienced any new or additional symptoms?',
            'placeholder': 'Describe any new symptoms, or write "None" if no new symptoms',
            'required': False
        })
        
        # Medication changes
        questions.append({
            'id': 'medication_status',
            'type': 'select',
            'label': 'Are you still taking the medication that caused the side effect?',
            'options': ['Yes, still taking', 'Stopped taking', 'Switched to alternative', 'Dosage changed'],
            'required': True
        })
        
        # Additional notes
        questions.append({
            'id': 'additional_notes',
            'type': 'textarea',
            'label': 'Any additional information you would like to share?',
            'placeholder': 'Optional: Share any other relevant information',
            'required': False
        })
        
        return questions
    
    def create_email_html(self, patient, followup_token: str) -> str:
        """Generate the HTML email content with the follow-up form link"""
        
        form_url = f"{APP_BASE_URL}/followup/{patient.id}/{followup_token}"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
        .content {{ background: #f9fafb; padding: 30px; border: 1px solid #e5e7eb; }}
        .button {{ display: inline-block; background: #3B82F6; color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: 600; margin: 20px 0; }}
        .button:hover {{ background: #2563eb; }}
        .footer {{ background: #1f2937; color: #9ca3af; padding: 20px; border-radius: 0 0 10px 10px; text-align: center; font-size: 12px; }}
        .info-box {{ background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #3B82F6; }}
        .highlight {{ color: #3B82F6; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0;">📋 Follow-up Request</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">Pharmacovigilance Safety Monitoring</p>
        </div>
        
        <div class="content">
            <p>Dear <span class="highlight">{patient.name}</span>,</p>
            
            <p>As part of our ongoing safety monitoring, we would like to follow up on the adverse event report 
            you submitted regarding <strong>{patient.drug_name}</strong>.</p>
            
            <div class="info-box">
                <p style="margin: 0;"><strong>📅 Original Report Date:</strong> {patient.created_at.strftime('%B %d, %Y') if patient.created_at else 'N/A'}</p>
                <p style="margin: 10px 0 0 0;"><strong>💊 Medication:</strong> {patient.drug_name}</p>
                <p style="margin: 10px 0 0 0;"><strong>📝 Reported Symptoms:</strong> {patient.symptoms or 'Not specified'}</p>
            </div>
            
            <p>Your feedback is crucial for ensuring medication safety. Please take a few minutes to complete our follow-up questionnaire:</p>
            
            <div style="text-align: center;">
                <a href="{form_url}" class="button">Complete Follow-up Form →</a>
            </div>
            
            <p style="color: #6b7280; font-size: 14px;">
                ⏰ This link will expire in <strong>7 days</strong>.<br>
                🔒 Your information is kept confidential and secure.
            </p>
        </div>
        
        <div class="footer">
            <p>This is an automated message from the Pharmacovigilance Safety Monitoring System.</p>
            <p>If you did not submit a report or have questions, please contact your healthcare provider.</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def create_email_text(self, patient, followup_token: str) -> str:
        """Generate plain text version of the email"""
        form_url = f"{APP_BASE_URL}/followup/{patient.id}/{followup_token}"
        
        text = f"""
FOLLOW-UP REQUEST - Pharmacovigilance Safety Monitoring

Dear {patient.name},

As part of our ongoing safety monitoring, we would like to follow up on the adverse event report you submitted regarding {patient.drug_name}.

Original Report Details:
- Report Date: {patient.created_at.strftime('%B %d, %Y') if patient.created_at else 'N/A'}
- Medication: {patient.drug_name}
- Reported Symptoms: {patient.symptoms or 'Not specified'}

Your feedback is crucial for ensuring medication safety. Please complete our follow-up questionnaire by visiting:

{form_url}

This link will expire in 7 days.

Your information is kept confidential and secure.

---
This is an automated message from the Pharmacovigilance Safety Monitoring System.
If you did not submit a report or have questions, please contact your healthcare provider.
"""
        return text
    
    def send_followup_email(self, patient, followup_token: str = None) -> Dict[str, Any]:
        """
        Send a follow-up questionnaire email to the patient.
        
        Args:
            patient: Patient model object
            followup_token: Optional pre-generated token, will generate if not provided
            
        Returns:
            Dict with success status and details
        """
        # Validate email configuration
        if not self.is_email_configured():
            return {
                'success': False,
                'error': 'Email not configured. Please set GMAIL_ADDRESS and GMAIL_APP_PASSWORD environment variables.',
                'setup_required': True
            }
        
        # Validate patient has email
        if not patient.email:
            return {
                'success': False,
                'error': f'Patient {patient.id} does not have an email address on file.',
                'patient_id': patient.id
            }
        
        # Generate token if not provided
        if not followup_token:
            followup_token = self.generate_followup_token(patient.id)
        
        try:
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f'📋 Follow-up Request: {patient.drug_name} Safety Monitoring'
            msg['From'] = self.email_config['email']
            msg['To'] = patient.email
            
            # Attach both plain text and HTML versions
            text_content = self.create_email_text(patient, followup_token)
            html_content = self.create_email_html(patient, followup_token)
            
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email via Gmail SMTP
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['email'], self.email_config['password'])
                server.send_message(msg)
            
            return {
                'success': True,
                'message': f'Follow-up email sent successfully to {patient.email}',
                'patient_id': patient.id,
                'token': followup_token,
                'sent_at': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(days=7)).isoformat()
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                'success': False,
                'error': 'Gmail authentication failed. Please check your email and app password.',
                'help': 'Make sure you are using an App Password, not your regular Gmail password.'
            }
        except smtplib.SMTPException as e:
            return {
                'success': False,
                'error': f'SMTP error: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to send email: {str(e)}'
            }
    
    def create_whatsapp_message(self, patient, followup_token: str) -> str:
        """Generate the WhatsApp message content with the follow-up form link"""
        form_url = f"{APP_BASE_URL}/followup/{patient.id}/{followup_token}"
        
        message = f"""📋 *Follow-up Request*
_Pharmacovigilance Safety Monitoring_

Dear *{patient.name}*,

We would like to follow up on your adverse event report regarding *{patient.drug_name}*.

📅 *Report Date:* {patient.created_at.strftime('%B %d, %Y') if patient.created_at else 'N/A'}
💊 *Medication:* {patient.drug_name}
📝 *Symptoms:* {patient.symptoms or 'Not specified'}

Please take a few minutes to complete our follow-up questionnaire:
👉 {form_url}

⏰ This link expires in 7 days.
🔒 Your information is kept confidential.

Reply *HELP* if you have questions or *STOP* to opt out.

_This is an automated message from the Pharmacovigilance Safety System._"""
        
        return message
    
    def send_followup_whatsapp(self, patient, followup_token: str = None) -> Dict[str, Any]:
        """
        Send a follow-up questionnaire via WhatsApp to the patient.
        
        Args:
            patient: Patient model object with phone number
            followup_token: Optional pre-generated token, will generate if not provided
            
        Returns:
            Dict with success status and details
        """
        # Validate WhatsApp configuration
        if not self.is_whatsapp_configured():
            return {
                'success': False,
                'error': 'WhatsApp not configured. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_FROM environment variables.',
                'setup_required': True
            }
        
        # Validate patient has phone number
        if not patient.phone:
            return {
                'success': False,
                'error': f'Patient {patient.id} does not have a phone number on file.',
                'patient_id': patient.id
            }
        
        # Ensure Twilio client is initialized
        if not self.twilio_client:
            return {
                'success': False,
                'error': 'Twilio client not initialized. Check your credentials.',
                'setup_required': True
            }
        
        # Generate token if not provided
        if not followup_token:
            followup_token = self.generate_followup_token(patient.id)
        
        try:
            # Format phone number for WhatsApp (needs to be in international format)
            phone = patient.phone.strip()
            if not phone.startswith('+'):
                # Assume Indian number if no country code
                if phone.startswith('0'):
                    phone = '+91' + phone[1:]
                elif len(phone) == 10:
                    phone = '+91' + phone
                else:
                    phone = '+' + phone
            
            # Create WhatsApp message
            message_body = self.create_whatsapp_message(patient, followup_token)
            
            # Send via Twilio
            message = self.twilio_client.messages.create(
                from_=self.twilio_config['whatsapp_from'],
                body=message_body,
                to=f'whatsapp:{phone}'
            )
            
            return {
                'success': True,
                'message': f'Follow-up WhatsApp sent successfully to {phone}',
                'patient_id': patient.id,
                'phone': phone,
                'message_sid': message.sid,
                'token': followup_token,
                'sent_at': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(days=7)).isoformat()
            }
            
        except Exception as e:
            error_message = str(e)
            if 'unverified' in error_message.lower():
                return {
                    'success': False,
                    'error': f'WhatsApp number not verified. In Twilio Sandbox mode, the recipient must first message your Twilio number.',
                    'help': 'Ask the patient to send "join <sandbox-keyword>" to your Twilio WhatsApp number first.',
                    'phone': patient.phone
                }
            return {
                'success': False,
                'error': f'Failed to send WhatsApp: {error_message}',
                'phone': patient.phone
            }
    
    def send_conversational_whatsapp(self, patient, tracking=None) -> Dict[str, Any]:
        """
        Start a conversational WhatsApp flow with the patient.
        Instead of sending a link, this starts an interactive chat where
        questions are asked one by one.
        
        Args:
            patient: Patient model object with phone number
            tracking: Optional AgentFollowupTracking record
            
        Returns:
            Dict with success status and details
        """
        # Validate WhatsApp configuration
        if not self.is_whatsapp_configured():
            return {
                'success': False,
                'error': 'WhatsApp not configured. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_FROM environment variables.',
                'setup_required': True
            }
        
        # Validate patient has phone number
        if not patient.phone:
            return {
                'success': False,
                'error': f'Patient {patient.id} does not have a phone number on file.',
                'patient_id': patient.id
            }
        
        # Ensure Twilio client is initialized
        if not self.twilio_client:
            return {
                'success': False,
                'error': 'Twilio client not initialized. Check your credentials.',
                'setup_required': True
            }
        
        try:
            # Format phone number for WhatsApp
            phone = patient.phone.strip()
            if not phone.startswith('+'):
                if phone.startswith('0'):
                    phone = '+91' + phone[1:]
                elif len(phone) == 10:
                    phone = '+91' + phone
                else:
                    phone = '+' + phone
            
            # Create conversational welcome message
            message_body = f"""🙏 *Hello {patient.name}!*

Welcome to our Patient Care Program.

We're reaching out regarding your report about *{patient.drug_name}*. We'd like to check on your wellbeing with a few quick questions.

📋 *Please select your preferred language:*

1️⃣ English
2️⃣ Hindi
3️⃣ Bengali
4️⃣ Telugu
5️⃣ Marathi
6️⃣ Tamil
7️⃣ Gujarati
8️⃣ Kannada
9️⃣ Malayalam
🔟 Punjabi

_Reply with the number (1-10) or language name._

🔒 Your responses are confidential and help us ensure your safety."""
            
            # Send via Twilio
            message = self.twilio_client.messages.create(
                from_=self.twilio_config['whatsapp_from'],
                body=message_body,
                to=f'whatsapp:{phone}'
            )
            
            # Update tracking to awaiting language state if provided
            if tracking:
                from models import db
                tracking.chatbot_state = 'awaiting_language'
                tracking.last_question_sent_at = datetime.utcnow()
                db.session.commit()
            
            return {
                'success': True,
                'message': f'Conversational WhatsApp started with {phone}',
                'patient_id': patient.id,
                'phone': phone,
                'message_sid': message.sid,
                'conversation_started': True,
                'sent_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_message = str(e)
            if 'unverified' in error_message.lower():
                return {
                    'success': False,
                    'error': f'WhatsApp number not verified. In Twilio Sandbox mode, the recipient must first message your Twilio number.',
                    'help': 'Ask the patient to send "join <sandbox-keyword>" to your Twilio WhatsApp number first.',
                    'phone': patient.phone
                }
            return {
                'success': False,
                'error': f'Failed to send WhatsApp: {error_message}',
                'phone': patient.phone
            }
    
    def send_followup_all_channels(self, patient, followup_token: str = None) -> Dict[str, Any]:
        """
        Send follow-up via all available channels (email and WhatsApp).
        
        Args:
            patient: Patient model object
            followup_token: Optional pre-generated token
            
        Returns:
            Dict with results from each channel
        """
        if not followup_token:
            followup_token = self.generate_followup_token(patient.id)
        
        results = {
            'patient_id': patient.id,
            'token': followup_token,
            'email': None,
            'whatsapp': None,
            'channels_sent': 0
        }
        
        # Try email if patient has email
        if patient.email:
            results['email'] = self.send_followup_email(patient, followup_token)
            if results['email'].get('success'):
                results['channels_sent'] += 1
        else:
            results['email'] = {'success': False, 'reason': 'no_email'}
        
        # Try WhatsApp if patient has phone
        if patient.phone:
            results['whatsapp'] = self.send_followup_whatsapp(patient, followup_token)
            if results['whatsapp'].get('success'):
                results['channels_sent'] += 1
        else:
            results['whatsapp'] = {'success': False, 'reason': 'no_phone'}
        
        results['success'] = results['channels_sent'] > 0
        
        return results
    
    def process_followup_response(self, patient, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the follow-up form response and update patient data.
        This will also recalculate the case scoring.
        
        Args:
            patient: Patient model object
            response_data: Dictionary containing form responses
            
        Returns:
            Dict with update status and new scoring
        """
        from pv_backend.services.case_scoring import evaluate_case, score_case, check_followup
        
        updates_made = []
        
        try:
            # Update symptom onset date if provided
            if response_data.get('symptom_onset_date'):
                patient.symptom_onset_date = datetime.strptime(
                    response_data['symptom_onset_date'], '%Y-%m-%d'
                )
                updates_made.append('symptom_onset_date')
            
            # Update symptom resolution date if provided
            if response_data.get('symptom_resolution_date'):
                patient.symptom_resolution_date = datetime.strptime(
                    response_data['symptom_resolution_date'], '%Y-%m-%d'
                )
                patient.has_clear_timeline = True
                updates_made.append('symptom_resolution_date')
            
            # Update doctor confirmation if they visited
            if response_data.get('doctor_visit') == 'Yes':
                patient.doctor_confirmed = True
                patient.doctor_confirmation_date = datetime.utcnow()
                updates_made.append('doctor_confirmed')
            
            # Update current status/severity
            if response_data.get('current_status'):
                status_mapping = {
                    'Fully Recovered': 'Recovered',
                    'Improving': 'Recovering',
                    'Same as before': 'Not recovered',
                    'Worsening': 'Not recovered',
                    'New symptoms appeared': 'Not recovered'
                }
                # Could add an outcome field update here
                updates_made.append('current_status')
            
            # Update severity based on response
            if response_data.get('current_severity'):
                severity_mapping = {
                    'None': 'Low',
                    'Mild': 'Low',
                    'Moderate': 'Medium',
                    'Severe': 'High'
                }
                patient.risk_level = severity_mapping.get(response_data['current_severity'], patient.risk_level)
                updates_made.append('risk_level')
            
            # Append additional symptoms
            if response_data.get('additional_symptoms') and response_data['additional_symptoms'].lower() != 'none':
                if patient.symptoms:
                    patient.symptoms = f"{patient.symptoms}\n\n[Follow-up {datetime.utcnow().strftime('%Y-%m-%d')}]: {response_data['additional_symptoms']}"
                else:
                    patient.symptoms = response_data['additional_symptoms']
                updates_made.append('symptoms')
            
            # Record follow-up response
            patient.followup_response_date = datetime.utcnow()
            patient.followup_responded = True
            updates_made.append('followup_response')
            
            # Store the scoring before recalculation
            old_score = patient.case_score
            old_followup_score = patient.followup_responsiveness_score
            
            # Recalculate case scoring (Steps 7-9)
            evaluate_case(patient)
            score_case(patient)
            check_followup(patient)
            
            # Calculate score improvement
            score_change = (patient.case_score or 0) - (old_score or 0)
            followup_score_change = (patient.followup_responsiveness_score or 0) - (old_followup_score or 0)
            
            return {
                'success': True,
                'message': 'Follow-up response processed successfully',
                'patient_id': patient.id,
                'updates_made': updates_made,
                'scoring': {
                    'previous_case_score': old_score,
                    'new_case_score': patient.case_score,
                    'score_change': score_change,
                    'previous_followup_score': old_followup_score,
                    'new_followup_score': patient.followup_responsiveness_score,
                    'followup_score_change': followup_score_change,
                    'strength_level': patient.strength_level,
                    'completeness_score': round(patient.completeness_score * 100, 1) if patient.completeness_score else 0,
                    'interpretation': patient.case_score_interpretation
                },
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing follow-up response: {str(e)}',
                'patient_id': patient.id
            }
    
    def send_bulk_followup(self, patients: list) -> Dict[str, Any]:
        """
        Send follow-up emails to multiple patients who need follow-up.
        
        Args:
            patients: List of Patient model objects
            
        Returns:
            Summary of sent emails
        """
        results = {
            'total': len(patients),
            'sent': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        for patient in patients:
            if not patient.email:
                results['skipped'] += 1
                results['details'].append({
                    'patient_id': patient.id,
                    'status': 'skipped',
                    'reason': 'No email address'
                })
                continue
            
            result = self.send_followup_email(patient)
            
            if result['success']:
                results['sent'] += 1
            else:
                results['failed'] += 1
            
            results['details'].append({
                'patient_id': patient.id,
                'status': 'sent' if result['success'] else 'failed',
                'message': result.get('message') or result.get('error')
            })
        
        return results


# Utility function for easy import
def create_followup_agent(db_session=None) -> FollowupAgent:
    """Factory function to create a FollowupAgent instance"""
    return FollowupAgent(db_session)


# ============================================================================
# PV AGENT ORCHESTRATOR - Day 1/3/5/7 Follow-up Cycle
# ============================================================================

class PVAgentOrchestrator:
    """
    Enhanced Pharmacovigilance Agent that manages the complete follow-up cycle.
    
    Workflow:
    1. On new patient: Create tracking record, run case scoring, get LLM questions
    2. Day 1, 3, 5, 7: Process previous responses, re-score, get new questions
    3. Each day: Start conversational WhatsApp (ignoring email for now)
    4. If patient says "fine": Remove from tracking
    
    Question Flow per Day:
    - Day 1: Predefined wellness/symptom questions + Gemini-generated questions based on case scoring
    - Day 3: Predefined progression questions + Gemini-generated follow-up questions
    - Day 5: Predefined clinical impact questions + Gemini-generated personalized questions
    - Day 7: Predefined resolution questions + Gemini-generated final assessment questions
    """
    
    FOLLOWUP_DAYS = [1, 3, 5, 7]
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.followup_agent = FollowupAgent(db_session)
    
    def start_tracking(self, patient) -> dict:
        """
        Start tracking a new patient for Day 1/3/5/7 follow-up cycle.
        Called when a new patient is created (database trigger).
        
        This triggers the WhatsApp conversation with Day 1 questions:
        - Predefined Day 1 questions
        - Gemini-generated personalized questions based on case scoring
        
        Returns:
            Dict with tracking info and initial questions
        """
        from models import db, AgentFollowupTracking
        from .case_scoring import evaluate_case, score_case
        from .llm_service import get_combined_questions, get_day_specific_questions
        
        try:
            # 1. Run initial case scoring (this will be used by Gemini for personalization)
            evaluate_case(patient)
            score_case(patient)
            print(f"📊 Case scoring complete: Score={patient.case_score}, Strength={patient.strength_level}")
            
            # 2. Get Day 1 combined questions (predefined + LLM personalized)
            questions_result = get_combined_questions(patient, previous_responses=None, current_day=1)
            
            # 3. Create tracking record with Day 1 questions
            tracking = AgentFollowupTracking(
                patient_id=patient.id,
                current_day=1,
                next_followup_date=datetime.utcnow() + timedelta(days=2),  # Day 3 is 2 days from Day 1
                predefined_questions=questions_result.get('predefined_questions', []),
                llm_questions=questions_result.get('llm_questions', []),
                unanswered_questions=questions_result.get('all_questions', []),
                day1_case_score=patient.case_score
            )
            
            db.session.add(tracking)
            db.session.commit()
            
            print(f"✅ Started PV Agent tracking for patient {patient.id}")
            print(f"📋 Day 1 Questions: {len(questions_result.get('predefined_questions', []))} predefined + {len(questions_result.get('llm_questions', []))} Gemini-generated")
            
            # 4. Send Day 1 WhatsApp message (starting conversational flow)
            send_result = self._send_day_messages(patient, tracking, 1)
            
            return {
                'success': True,
                'tracking_id': tracking.id,
                'current_day': 1,
                'predefined_questions': questions_result.get('predefined_questions', []),
                'llm_questions': questions_result.get('llm_questions', []),
                'all_questions': questions_result.get('all_questions', []),
                'analysis': questions_result.get('analysis', ''),
                'send_result': send_result
            }
            
        except Exception as e:
            print(f"❌ Error starting PV tracking: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def process_day_cycle(self, tracking_id: int) -> dict:
        """
        Process a scheduled day cycle (Day 3, 5, or 7).
        1. Process previous day's responses
        2. Re-run case scoring (used by Gemini for personalization)
        3. Get new day-specific questions (predefined + Gemini personalized)
        4. Send WhatsApp conversation
        
        Returns:
            Dict with cycle results
        """
        from models import db, AgentFollowupTracking, Patient
        from .case_scoring import evaluate_case, score_case
        from .llm_service import get_combined_questions
        
        tracking = AgentFollowupTracking.query.get(tracking_id)
        if not tracking or tracking.status != 'active':
            return {'success': False, 'error': 'Tracking not found or inactive'}
        
        patient = Patient.query.get(tracking.patient_id)
        if not patient:
            return {'success': False, 'error': 'Patient not found'}
        
        # Get previous responses for Gemini context
        previous_responses = self._get_previous_responses(tracking)
        
        # Re-run case scoring (this updated score is sent to Gemini for personalization)
        evaluate_case(patient)
        score_case(patient)
        print(f"📊 Updated case scoring: Score={patient.case_score}, Strength={patient.strength_level}")
        
        # Move to next day first to get correct questions
        current_day = tracking.current_day
        next_day_idx = self.FOLLOWUP_DAYS.index(current_day) + 1
        
        if next_day_idx >= len(self.FOLLOWUP_DAYS):
            # Completed all days
            tracking.status = 'completed'
            db.session.commit()
            return {'success': True, 'message': 'All follow-up days completed', 'completed': True}
        
        next_day = self.FOLLOWUP_DAYS[next_day_idx]
        tracking.current_day = next_day
        
        # Get day-specific questions (predefined + Gemini personalized based on case scoring)
        questions_result = get_combined_questions(patient, previous_responses, current_day=next_day)
        
        # Update tracking with new day-specific questions
        tracking.predefined_questions = questions_result.get('predefined_questions', [])
        tracking.llm_questions = questions_result.get('llm_questions', [])
        tracking.unanswered_questions = questions_result.get('all_questions', [])
        tracking.current_question_index = 0
        tracking.reminder_count = 0
        
        # Update case score for this day
        setattr(tracking, f'day{next_day}_case_score', patient.case_score)
        
        db.session.commit()
        
        print(f"📋 Day {next_day} Questions: {len(questions_result.get('predefined_questions', []))} predefined + {len(questions_result.get('llm_questions', []))} Gemini-generated")
        
        # Send WhatsApp for this day
        send_result = self._send_day_messages(patient, tracking, next_day)
        
        return {
            'success': True,
            'current_day': next_day,
            'previous_day': current_day,
            'case_score': patient.case_score,
            'strength_level': patient.strength_level,
            'predefined_questions': questions_result.get('predefined_questions', []),
            'llm_questions': questions_result.get('llm_questions', []),
            'all_questions': questions_result.get('all_questions', []),
            'analysis': questions_result.get('analysis', ''),
            'send_result': send_result
        }
    
    def process_response(self, tracking_id: int, responses: dict, channel: str = 'email') -> dict:
        """
        Process patient response for current day.
        
        Args:
            tracking_id: AgentFollowupTracking ID
            responses: Dict of question_id -> answer
            channel: 'email' or 'whatsapp'
            
        Returns:
            Dict with processing results
        """
        from models import db, AgentFollowupTracking, Patient
        from .llm_service import PrivacySafeLLMService
        
        tracking = AgentFollowupTracking.query.get(tracking_id)
        if not tracking:
            return {'success': False, 'error': 'Tracking not found'}
        
        patient = Patient.query.get(tracking.patient_id)
        current_day = tracking.current_day
        
        # Mark channel as responded
        if channel == 'email':
            setattr(tracking, f'day{current_day}_email_responded', True)
        
        # Store responses
        setattr(tracking, f'day{current_day}_responses', responses)
        
        # Validate responses with LLM and update patient
        llm_service = PrivacySafeLLMService()
        updated_columns = []
        
        for question_id, answer in responses.items():
            # Check if patient says they're fine
            intent = llm_service.detect_patient_intent(str(answer))
            if intent == 'fine':
                tracking.patient_said_fine = True
                tracking.status = 'patient_fine'
                db.session.commit()
                print(f"✅ Patient {patient.id} said they're fine - stopping follow-ups")
                return {
                    'success': True,
                    'patient_fine': True,
                    'message': 'Patient reported feeling fine - follow-up stopped'
                }
            
            # Validate and map response
            all_questions = (tracking.predefined_questions or []) + (tracking.llm_questions or [])
            question_data = next((q for q in all_questions if q.get('id') == question_id or q.get('question') == question_id), None)
            
            if question_data and question_data.get('maps_to_column'):
                column = question_data['maps_to_column']
                validation = llm_service.validate_response(
                    question_data.get('question', question_id),
                    column,
                    str(answer)
                )
                
                if validation.get('is_useful') and validation.get('extracted_value') is not None:
                    if hasattr(patient, column):
                        setattr(patient, column, validation['extracted_value'])
                        updated_columns.append(column)
        
        db.session.commit()
        
        return {
            'success': True,
            'updated_columns': updated_columns,
            'patient_fine': False,
            'next_day': self._get_next_day(current_day)
        }
    
    def should_send_whatsapp(self, tracking_id: int) -> bool:
        """
        Check if WhatsApp should be sent for current day.
        Returns False if email form was already filled for this day.
        """
        from models import AgentFollowupTracking
        
        tracking = AgentFollowupTracking.query.get(tracking_id)
        if not tracking:
            return False
        
        current_day = tracking.current_day
        email_responded = getattr(tracking, f'day{current_day}_email_responded', False)
        
        # Don't send WhatsApp if email was already responded
        return not email_responded
    
    def _send_day_messages(self, patient, tracking, day: int) -> dict:
        """
        Send messages for a specific day using DUAL CHANNELS.
        
        CHANNEL LOGIC (Day 1/3/5/7):
        1. Send EMAIL with follow-up form link
        2. Send WhatsApp conversational flow (question-by-question)
        3. Patient responds via either channel
        4. Once responded via ANY channel → Don't ask same day questions again via other channel
        5. Move to next day questions
        
        This ensures patient can respond via their preferred channel.
        """
        from models import db, FollowupToken
        from pv_backend.routes.followup_routes import store_followup_token
        
        results = {
            'email': None, 
            'whatsapp': None,
            'routing_decision': None,
            'day': day,
            'channels_sent': 0
        }
        
        # Generate token (for email and WhatsApp forms)
        token = self.followup_agent.generate_followup_token(patient.id)
        
        # Store the token in the database
        store_followup_token(patient.id, token, expires_in_days=7)
        print(f"✅ Stored follow-up token for patient {patient.id}")
        
        # Check if patient already responded to this day
        day_responses_attr = f'day{day}_responses'
        has_responded = bool(getattr(tracking, day_responses_attr, None))
        
        if has_responded:
            print(f"📬 Day {day}: Patient already responded for this day - skipping dual sends")
            results['routing_decision'] = 'already_responded'
            return results
        
        # ===== CHANNEL 1: EMAIL =====
        if patient.email:
            email_result = self.followup_agent.send_followup_email(patient, token)
            results['email'] = email_result
            setattr(tracking, f'day{day}_email_sent', True)
            
            if email_result.get('success'):
                print(f"✅ Day {day}: Email sent to {patient.email}")
                results['channels_sent'] += 1
            else:
                print(f"⚠️ Day {day}: Email failed - {email_result.get('error')}")
        else:
            print(f"⚠️ Day {day}: No email for patient {patient.id}")
            results['email'] = {'success': False, 'error': 'No email address'}
        
        # ===== CHANNEL 2: WHATSAPP =====
        # Always send WhatsApp if phone number exists (dual channel approach)
        if patient.phone:
            # Use conversational WhatsApp that asks questions one by one
            # Questions are day-specific: predefined + Gemini-generated personalized questions
            whatsapp_result = self.followup_agent.send_conversational_whatsapp(patient, tracking)
            results['whatsapp'] = whatsapp_result
            setattr(tracking, f'day{day}_whatsapp_sent', True)
            
            if whatsapp_result.get('success'):
                print(f"✅ Day {day}: WhatsApp conversational chat started with {patient.phone}")
                print(f"   📋 Questions loaded: {len(tracking.predefined_questions or [])} predefined + {len(tracking.llm_questions or [])} Gemini-generated")
                results['channels_sent'] += 1
            else:
                print(f"⚠️ Day {day}: WhatsApp failed - {whatsapp_result.get('error')}")
        else:
            print(f"⚠️ Day {day}: No phone number for patient {patient.id}")
            results['whatsapp'] = {'success': False, 'error': 'No phone number'}
        
        # ===== ROUTING SUMMARY =====
        if results['channels_sent'] == 2:
            results['routing_decision'] = 'dual_channel_active'
            print(f"📊 Day {day}: DUAL CHANNEL MODE - Patient can respond via email OR WhatsApp")
            print(f"   💡 Once responded via either channel, other channel for this day is skipped")
        elif results['channels_sent'] == 1:
            if results['email'] and results['email'].get('success'):
                results['routing_decision'] = 'email_only'
                print(f"📊 Day {day}: EMAIL ONLY - WhatsApp unavailable")
            else:
                results['routing_decision'] = 'whatsapp_only'
                print(f"📊 Day {day}: WHATSAPP ONLY - Email unavailable")
        else:
            results['routing_decision'] = 'no_channels_available'
            print(f"💥 Day {day}: NO CONTACT CHANNELS AVAILABLE - Patient cannot be reached!")
        
        # ===== TRACKING UPDATES =====
        # Record message send time
        tracking.messages_sent_count = (tracking.messages_sent_count or 0) + results['channels_sent']
        tracking.last_response_received_at = datetime.utcnow()
        
        # Schedule next day
        next_day = self._get_next_day(day)
        if next_day:
            days_until_next = next_day - day
            tracking.next_followup_date = datetime.utcnow() + timedelta(days=days_until_next)
            print(f"📅 Next follow-up (Day {next_day}) scheduled in {days_until_next} days")
        
        db.session.commit()
        
        return results
    
    def _get_previous_responses(self, tracking) -> dict:
        """Get all previous day responses."""
        responses = {}
        for day in self.FOLLOWUP_DAYS:
            if day < tracking.current_day:
                day_responses = getattr(tracking, f'day{day}_responses', None)
                if day_responses:
                    responses[f'day{day}'] = day_responses
        return responses
    
    def _get_next_day(self, current_day: int) -> int:
        """Get the next scheduled day, or None if complete."""
        try:
            idx = self.FOLLOWUP_DAYS.index(current_day)
            if idx + 1 < len(self.FOLLOWUP_DAYS):
                return self.FOLLOWUP_DAYS[idx + 1]
        except ValueError:
            pass
        return None
    
    def get_active_trackings(self) -> list:
        """Get all active tracking records for scheduled processing."""
        from models import AgentFollowupTracking
        
        return AgentFollowupTracking.query.filter_by(status='active').all()
    
    def get_due_followups(self) -> list:
        """Get trackings that are due for next day cycle."""
        from models import AgentFollowupTracking
        
        now = datetime.utcnow()
        return AgentFollowupTracking.query.filter(
            AgentFollowupTracking.status == 'active',
            AgentFollowupTracking.next_followup_date <= now
        ).all()


def create_pv_orchestrator(db_session=None) -> PVAgentOrchestrator:
    """Factory function to create a PVAgentOrchestrator instance."""
    return PVAgentOrchestrator(db_session)


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("Follow-up Agent - Setup Guide")
    print("=" * 60)
    print("""
To use this agent, you need to configure Gmail SMTP:

1. ENABLE 2-FACTOR AUTHENTICATION:
   - Go to: https://myaccount.google.com/security
   - Enable 2-Step Verification

2. CREATE APP PASSWORD:
   - Go to: https://myaccount.google.com/apppasswords
   - Select 'Mail' and your device
   - Copy the 16-character password generated

3. SET ENVIRONMENT VARIABLES:
   
   Windows (PowerShell):
   $env:GMAIL_ADDRESS = "your-email@gmail.com"
   $env:GMAIL_APP_PASSWORD = "your-16-char-password"
   
   Windows (CMD):
   set GMAIL_ADDRESS=your-email@gmail.com
   set GMAIL_APP_PASSWORD=your-16-char-password
   
   Linux/Mac:
   export GMAIL_ADDRESS="your-email@gmail.com"
   export GMAIL_APP_PASSWORD="your-16-char-password"

4. ALTERNATIVELY, create a .env file:
   GMAIL_ADDRESS=your-email@gmail.com
   GMAIL_APP_PASSWORD=your-16-char-password

5. TEST THE CONFIGURATION:
   python -c "from pv_backend.services.followup_agent import FollowupAgent; agent = FollowupAgent(); print('Configured:', agent.is_email_configured())"
""")

