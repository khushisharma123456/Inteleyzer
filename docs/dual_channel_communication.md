# Dual-Channel Communication: WhatsApp + Email Form

## Summary

This feature enables patients to receive follow-up forms via **both WhatsApp and Email**. If a patient fills the form through email, the WhatsApp agent automatically recognizes this and thanks them instead of asking questions again.

## How It Works

```
Doctor adds patient with phone + email
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
WhatsApp starts    Email with form link sent
    ↓                   ↓
Questions asked    Patient fills form
    ↓                   ↓
    └─────────┬─────────┘
              ↓
     Check if form filled
              ↓
       ┌──────┴──────┐
       ↓             ↓
    Filled       Not Filled
       ↓             ↓
 "Thank you!"   Continue questions
```

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `email_service.py` | SMTP email sending, multi-language templates |
| `form_service.py` | Form token management, question definitions |
| `templates/patient_form.html` | Multi-language web form |

### Modified Files

| File | Changes |
|------|---------|
| `ConversationalAgent.py` | Added form-filled detection, email support |
| `agentBackend.py` | Added form endpoints, email sending on start |
| `dataset/schema.sql` | Added email column, Form_Submission table |
| `.env.example` | Added email/form configuration |

---

## API Endpoints

### Start Conversation (with email)

```bash
curl -X POST http://localhost:8000/start-conversation \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "phone_number": "+1234567890",
    "email": "patient@example.com",
    "patient_name": "John Doe",
    "language": "en"
  }'
```

Response includes:
- `email_sent`: true/false
- `form_url`: URL sent to patient

### Serve Form

```
GET /form/{token}?lang=en
```

Serves the multi-language patient form.

### Submit Form

```bash
curl -X POST http://localhost:8000/api/form/submit \
  -H "Content-Type: application/json" \
  -d '{
    "token": "form_token_here",
    "responses": {
      "language": "en",
      "medicine_started": "yes",
      "adherence": "once",
      "food_relation": "after",
      "overall_feeling": "better",
      "new_symptoms": "no",
      "safety_confirm": "confirmed"
    }
  }'
```

### Send Clarification Form

```bash
curl -X POST http://localhost:8000/send-clarification-form \
  -H "Content-Type: application/json" \
  -d '{
    "visit_id": 123,
    "patient_id": "P001",
    "email": "patient@example.com",
    "patient_name": "John",
    "missing_fields": ["symptom_description", "severity"],
    "language": "en"
  }'
```

---

## Configuration Required

Copy `.env.example` to `.env` and configure:

### 1. Email Configuration (Gmail SMTP)

```env
EMAIL_SERVICE=smtp
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password  # NOT your Gmail password!
SENDER_NAME=Pharmacovigilance Team
SENDER_EMAIL=your_email@gmail.com
HOSPITAL_NAME=Healthcare Center
```

#### Getting Gmail App Password:
1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable 2-Factor Authentication
3. Go to App Passwords
4. Generate a new app password for "Mail"
5. Use this 16-character password as `SMTP_PASSWORD`

### 2. Form Configuration

```env
FORM_BASE_URL=http://localhost:8000/form
FORM_SECRET_KEY=generate-a-random-secret-key
```

For production, change `FORM_BASE_URL` to your deployed URL.

### 3. Twilio Configuration

```env
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```

---

## Multi-Language Support

The form supports 5 languages:

| Code | Language | Native |
|------|----------|--------|
| `en` | English | English |
| `hi` | Hindi | हिंदी |
| `ta` | Tamil | தமிழ் |
| `te` | Telugu | తెలుగు |
| `ml` | Malayalam | മലയാളം |

When patient selects a language in the form, all questions and labels switch dynamically via JavaScript.

---

## Form Questions

The form includes these questions (same as WhatsApp flow):

1. **Medicine Started** - Have you started taking the medicine?
2. **Adherence** - How often are you taking it?
3. **Food Relation** - When do you take it (before/after food)?
4. **Overall Feeling** - How are you feeling since starting?
5. **New Symptoms** - Any new symptoms? (conditional questions follow)
6. **Symptom Description** - Describe symptoms (if yes)
7. **Onset** - When did symptoms start? (if yes)
8. **Severity** - How severe? (if yes)
9. **Body Parts** - Which body parts affected? (if yes)
10. **Safety Confirm** - Confirm understanding of safety guidelines

---

## Testing

### 1. Test Email Service
```bash
python email_service.py
```

### 2. Test Form Service
```bash
python form_service.py
```

### 3. Start Backend
```bash
python agentBackend.py
```

### 4. Test Form in Browser
1. Start the server
2. Open: http://localhost:8000/form/test-token
3. Switch languages using radio buttons
4. Submit form

---

## Database Changes

### New Columns in Patient Table
```sql
email VARCHAR(255)              -- Patient's email
preferred_language VARCHAR(10)  -- Language preference (en, hi, ta, te, ml)
```

### New Table: Form_Submission
```sql
CREATE TABLE Form_Submission (
    submission_id INT PRIMARY KEY AUTO_INCREMENT,
    visit_id INT NOT NULL,
    patient_id VARCHAR(50) NOT NULL,
    form_type VARCHAR(50),    -- 'initial' or 'clarification'
    form_token VARCHAR(100),
    form_url VARCHAR(500),
    language VARCHAR(10),
    sent_at TIMESTAMP,
    filled_at TIMESTAMP NULL,
    responses JSON,
    status VARCHAR(50)        -- 'pending', 'sent', 'filled', 'expired'
);
```

---

## Clarification Flow

When missing data is detected (by Agent 2 - Data Quality):

1. API call to `/send-clarification-form`
2. Email sent with only the missing questions
3. Patient fills clarification form
4. WhatsApp recognizes and thanks patient

---

## Security Notes

- Form tokens are unique and expire after use
- Patients can only fill their own forms
- All form submissions are logged
- HTTPS recommended for production
