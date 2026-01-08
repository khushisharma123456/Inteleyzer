"""
📧 EMAIL SERVICE FOR PHARMACOVIGILANCE FOLLOW-UP

This module handles sending emails with form links to patients.
Supports Gmail SMTP and can be extended to use SendGrid or other services.

Author: Generated for dual-channel communication feature
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# EMAIL CONFIGURATION
# =============================================================================

EMAIL_SERVICE = os.getenv('EMAIL_SERVICE', 'smtp')  # 'smtp' or 'sendgrid'
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
SENDER_NAME = os.getenv('SENDER_NAME', 'Pharmacovigilance Team')
SENDER_EMAIL = os.getenv('SENDER_EMAIL', SMTP_USER)

# Form URL configuration
FORM_BASE_URL = os.getenv('FORM_BASE_URL', 'http://localhost:8000/form')

# Hospital/Clinic name for email templates
HOSPITAL_NAME = os.getenv('HOSPITAL_NAME', 'Healthcare Center')


# =============================================================================
# EMAIL TEMPLATES
# =============================================================================

def get_initial_form_email_html(patient_name: str, form_url: str, language: str = 'en') -> str:
    """
    Generate HTML email for initial form.
    
    Args:
        patient_name: Patient's name
        form_url: URL to the form
        language: Language code for the email
        
    Returns:
        HTML email content
    """
    # Multi-language subject and content
    content = {
        'en': {
            'greeting': f'Dear {patient_name},',
            'intro': 'Your doctor has requested a follow-up regarding your recent prescription.',
            'action': 'Please fill out this short form to help us monitor your health:',
            'button': 'Fill Follow-up Form',
            'whatsapp_note': 'You can also respond via WhatsApp if you prefer.',
            'regards': 'Best regards,',
            'team': f'{HOSPITAL_NAME} Pharmacovigilance Team'
        },
        'hi': {
            'greeting': f'प्रिय {patient_name},',
            'intro': 'आपके डॉक्टर ने आपके हाल के प्रिस्क्रिप्शन के संबंध में फॉलो-अप का अनुरोध किया है।',
            'action': 'कृपया अपने स्वास्थ्य की निगरानी में मदद के लिए यह छोटा फॉर्म भरें:',
            'button': 'फॉलो-अप फॉर्म भरें',
            'whatsapp_note': 'आप चाहें तो WhatsApp के माध्यम से भी जवाब दे सकते हैं।',
            'regards': 'सादर,',
            'team': f'{HOSPITAL_NAME} फार्माकोविजिलेंस टीम'
        },
        'ta': {
            'greeting': f'அன்புள்ள {patient_name},',
            'intro': 'உங்கள் சமீபத்திய மருந்து குறிப்பு தொடர்பாக உங்கள் மருத்துவர் பின்தொடர்தலைக் கோரியுள்ளார்.',
            'action': 'உங்கள் ஆரோக்கியத்தைக் கண்காணிக்க இந்த சிறிய படிவத்தை நிரப்பவும்:',
            'button': 'படிவத்தை நிரப்பு',
            'whatsapp_note': 'நீங்கள் விரும்பினால் WhatsApp வழியாகவும் பதிலளிக்கலாம்.',
            'regards': 'அன்புடன்,',
            'team': f'{HOSPITAL_NAME} மருந்து கண்காணிப்பு குழு'
        }
    }
    
    # Default to English if language not found
    c = content.get(language, content['en'])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px 10px 0 0;
            }}
            .content {{
                background: #f9f9f9;
                padding: 30px;
                border-radius: 0 0 10px 10px;
            }}
            .button {{
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white !important;
                padding: 15px 30px;
                text-decoration: none;
                border-radius: 25px;
                margin: 20px 0;
                font-weight: bold;
            }}
            .note {{
                background: #e8f4f8;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                font-size: 14px;
            }}
            .footer {{
                text-align: center;
                margin-top: 20px;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>💊 Medication Follow-up</h1>
        </div>
        <div class="content">
            <p>{c['greeting']}</p>
            <p>{c['intro']}</p>
            <p>{c['action']}</p>
            <p style="text-align: center;">
                <a href="{form_url}" class="button">{c['button']}</a>
            </p>
            <div class="note">
                📱 {c['whatsapp_note']}
            </div>
            <p style="margin-top: 30px;">
                {c['regards']}<br>
                <strong>{c['team']}</strong>
            </p>
        </div>
        <div class="footer">
            <p>This is an automated message from {HOSPITAL_NAME}</p>
        </div>
    </body>
    </html>
    """
    return html


def get_clarification_email_html(patient_name: str, form_url: str, 
                                  missing_fields: list, language: str = 'en') -> str:
    """
    Generate HTML email for clarification form.
    
    Args:
        patient_name: Patient's name
        form_url: URL to the clarification form
        missing_fields: List of fields that need clarification
        language: Language code
        
    Returns:
        HTML email content
    """
    content = {
        'en': {
            'greeting': f'Dear {patient_name},',
            'intro': 'We need some additional information to complete your follow-up.',
            'missing_intro': 'The following information is needed:',
            'action': 'Please fill out this short form:',
            'button': 'Provide Information',
            'thanks': 'Thank you for your cooperation.',
            'regards': 'Best regards,',
            'team': f'{HOSPITAL_NAME} Pharmacovigilance Team'
        },
        'hi': {
            'greeting': f'प्रिय {patient_name},',
            'intro': 'आपके फॉलो-अप को पूरा करने के लिए हमें कुछ अतिरिक्त जानकारी चाहिए।',
            'missing_intro': 'निम्नलिखित जानकारी आवश्यक है:',
            'action': 'कृपया यह छोटा फॉर्म भरें:',
            'button': 'जानकारी प्रदान करें',
            'thanks': 'आपके सहयोग के लिए धन्यवाद।',
            'regards': 'सादर,',
            'team': f'{HOSPITAL_NAME} फार्माकोविजिलेंस टीम'
        }
    }
    
    c = content.get(language, content['en'])
    
    # Generate missing fields list
    missing_html = "\n".join([f"<li>{field}</li>" for field in missing_fields])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px 10px 0 0;
            }}
            .content {{
                background: #f9f9f9;
                padding: 30px;
                border-radius: 0 0 10px 10px;
            }}
            .button {{
                display: inline-block;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white !important;
                padding: 15px 30px;
                text-decoration: none;
                border-radius: 25px;
                margin: 20px 0;
                font-weight: bold;
            }}
            .missing-list {{
                background: #fff3cd;
                padding: 15px 15px 15px 35px;
                border-radius: 8px;
                border-left: 4px solid #ffc107;
            }}
            .footer {{
                text-align: center;
                margin-top: 20px;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📋 Additional Information Needed</h1>
        </div>
        <div class="content">
            <p>{c['greeting']}</p>
            <p>{c['intro']}</p>
            <p><strong>{c['missing_intro']}</strong></p>
            <ul class="missing-list">
                {missing_html}
            </ul>
            <p>{c['action']}</p>
            <p style="text-align: center;">
                <a href="{form_url}" class="button">{c['button']}</a>
            </p>
            <p>{c['thanks']}</p>
            <p style="margin-top: 30px;">
                {c['regards']}<br>
                <strong>{c['team']}</strong>
            </p>
        </div>
        <div class="footer">
            <p>This is an automated message from {HOSPITAL_NAME}</p>
        </div>
    </body>
    </html>
    """
    return html


# =============================================================================
# EMAIL SENDING FUNCTIONS
# =============================================================================

def send_email(to_email: str, subject: str, html_content: str) -> bool:
    """
    Send an email using configured SMTP settings.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML email body
        
    Returns:
        True if sent successfully, False otherwise
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("⚠️ Email credentials not configured. Email not sent.")
        logger.info(f"[TEST MODE] Would send email to {to_email}: {subject}")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg['To'] = to_email
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Send via SMTP
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"✅ Email sent to {to_email}: {subject}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")
        return False


def send_form_email(to_email: str, patient_name: str, visit_id: int, 
                    patient_id: str, language: str = 'en') -> bool:
    """
    Send the initial follow-up form email to a patient.
    
    Args:
        to_email: Patient's email address
        patient_name: Patient's name
        visit_id: Visit ID for tracking
        patient_id: Patient ID
        language: Preferred language code
        
    Returns:
        True if sent successfully
    """
    # Generate form URL with parameters
    form_url = f"{FORM_BASE_URL}/{visit_id}?patient_id={patient_id}&lang={language}"
    
    # Generate email content
    html_content = get_initial_form_email_html(patient_name, form_url, language)
    
    # Email subject (multi-language)
    subjects = {
        'en': f'Your Medication Follow-up Form - {HOSPITAL_NAME}',
        'hi': f'आपका दवा फॉलो-अप फॉर्म - {HOSPITAL_NAME}',
        'ta': f'உங்கள் மருந்து பின்தொடர்தல் படிவம் - {HOSPITAL_NAME}'
    }
    subject = subjects.get(language, subjects['en'])
    
    return send_email(to_email, subject, html_content)


def send_clarification_email(to_email: str, patient_name: str, visit_id: int,
                              patient_id: str, missing_fields: list,
                              language: str = 'en') -> bool:
    """
    Send a clarification form email for missing/unclear data.
    
    Args:
        to_email: Patient's email address
        patient_name: Patient's name
        visit_id: Visit ID
        patient_id: Patient ID
        missing_fields: List of fields needing clarification
        language: Preferred language code
        
    Returns:
        True if sent successfully
    """
    # Generate clarification form URL
    form_url = f"{FORM_BASE_URL}/clarification/{visit_id}?patient_id={patient_id}&lang={language}"
    
    # Generate email content
    html_content = get_clarification_email_html(
        patient_name, form_url, missing_fields, language
    )
    
    # Email subject
    subjects = {
        'en': f'Additional Information Needed - {HOSPITAL_NAME}',
        'hi': f'अतिरिक्त जानकारी आवश्यक - {HOSPITAL_NAME}',
        'ta': f'கூடுதல் தகவல் தேவை - {HOSPITAL_NAME}'
    }
    subject = subjects.get(language, subjects['en'])
    
    return send_email(to_email, subject, html_content)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_email(email: str) -> bool:
    """
    Basic email validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid format
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def get_email_status() -> Dict[str, Any]:
    """
    Get current email service configuration status.
    
    Returns:
        Dict with configuration status
    """
    return {
        "service": EMAIL_SERVICE,
        "configured": bool(SMTP_USER and SMTP_PASSWORD),
        "smtp_host": SMTP_HOST,
        "smtp_port": SMTP_PORT,
        "sender_email": SENDER_EMAIL,
        "form_base_url": FORM_BASE_URL
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📧 Email Service - Test Mode")
    print("=" * 60)
    
    # Check configuration
    status = get_email_status()
    print(f"\n📋 Configuration Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test email validation
    test_emails = ["test@example.com", "invalid-email", "user@domain.co.in"]
    print(f"\n📧 Email Validation:")
    for email in test_emails:
        valid = "✅" if validate_email(email) else "❌"
        print(f"  {email}: {valid}")
    
    # Preview email content
    print(f"\n📄 Preview Initial Form Email (English):")
    preview = get_initial_form_email_html("John Doe", "https://example.com/form/1")
    print(f"  Generated {len(preview)} characters of HTML")
    
    print("\n" + "=" * 60)
