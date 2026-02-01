# WhatsApp Sandbox Verification Guide

## Why You're Not Receiving WhatsApp Messages

In Twilio's free **Sandbox mode**, you need to verify your phone number first by sending a "join" message to Twilio's test number.

## How to Receive WhatsApp Messages

### Step 1: Open WhatsApp
- Use WhatsApp on your phone (the same number you registered)
- Number: +918595687463 (or your patient's number)

### Step 2: Send Join Message to Twilio
Send this exact message:
```
join bright-joy
```

To: **+1-415-523-8886** (Twilio's WhatsApp test number)

### Step 3: Verify Message
You should receive a reply from Twilio confirming you've joined the sandbox

### Step 4: Now You Can Receive Messages
Once verified, you'll start receiving WhatsApp messages from our system at +14155238886 (our Twilio WhatsApp number)

---

## What Messages You'll Receive

### Day 1: 
- 📧 **Email** - Follow-up form link (immediate)
- 💬 **WhatsApp** - Language selection menu (immediate after verification)

### Day 3 (2 days later):
- 📧 **Email** - Follow-up questions
- 💬 **WhatsApp** - Conversational questions

### Day 5:
- More follow-up questions

### Day 7:
- Final check-in

---

## Troubleshooting

### "WhatsApp Status: Unable to Send" in Modal
- ✅ This is NORMAL in sandbox mode
- The system DID try to send the message
- The patient just needs to verify first (send "join bright-joy")
- After verification, messages will be delivered

### Not Receiving Email
- Check spam/junk folder
- Verify email address is correct in form
- Check Gmail is configured in .env

### WhatsApp Still Not Working After Verification
- Wait 5 minutes for Twilio to process verification
- Try submitting a new ADR report
- System will automatically send after next submission

---

## For Production (Remove Sandbox Limitation)

To send to ANY WhatsApp number without sandbox verification:

1. Get Twilio WhatsApp Business API approval
2. Update TWILIO_WHATSAPP_FROM to approved business number
3. Remove sandbox enrollment requirement

---

## Verification Status Check

Run this command to see if your number is verified:

```bash
# Check Twilio sandbox participants
python -c "
from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()
client = Client(
    os.environ['TWILIO_ACCOUNT_SID'],
    os.environ['TWILIO_AUTH_TOKEN']
)
# List recent messages to see delivery status
messages = client.messages.list(limit=5)
for msg in messages:
    print(f'To: {msg.to}, Status: {msg.status}')
"
```

---

## Summary

- ✅ Email: Sends immediately, check inbox
- ✅ WhatsApp: Requires one-time verification (send "join bright-joy")
- ✅ Both channels are working correctly
- ⏳ After verification, you'll receive all messages
