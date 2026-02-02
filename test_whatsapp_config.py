#!/usr/bin/env python3
"""Test script to verify WhatsApp/Twilio configuration"""

import os
from dotenv import load_dotenv

print("=" * 60)
print("WHATSAPP/TWILIO CONFIGURATION TEST")
print("=" * 60)

# Load environment variables
print("\n1. Loading .env file...")
load_dotenv(override=True)

# Check Gmail config
print("\n2. Gmail Configuration:")
gmail_address = os.environ.get('GMAIL_ADDRESS')
gmail_password = os.environ.get('GMAIL_APP_PASSWORD')
print(f"   GMAIL_ADDRESS: {gmail_address}")
print(f"   GMAIL_APP_PASSWORD: {'*' * len(gmail_password) if gmail_password else 'NOT SET'}")

# Check Twilio config
print("\n3. Twilio WhatsApp Configuration:")
twilio_sid = os.environ.get('TWILIO_ACCOUNT_SID')
twilio_token = os.environ.get('TWILIO_AUTH_TOKEN')
twilio_from = os.environ.get('TWILIO_WHATSAPP_FROM')

print(f"   TWILIO_ACCOUNT_SID: {twilio_sid}")
print(f"   TWILIO_AUTH_TOKEN: {twilio_token}")
print(f"   TWILIO_WHATSAPP_FROM: {twilio_from}")

# Check if Twilio is configured
is_configured = bool(twilio_sid and twilio_token and twilio_from)
print(f"\n4. WhatsApp Configured: {is_configured}")

# Try to import and initialize
print("\n5. Testing FollowupAgent initialization...")
try:
    from pv_backend.services.followup_agent import FollowupAgent
    agent = FollowupAgent()
    
    print(f"   Email configured: {agent.is_email_configured()}")
    print(f"   WhatsApp configured: {agent.is_whatsapp_configured()}")
    print(f"   Twilio client initialized: {agent.twilio_client is not None}")
    
    if agent.is_whatsapp_configured():
        print("   ✅ WhatsApp is ready to use!")
    else:
        print("   ❌ WhatsApp is NOT configured!")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
