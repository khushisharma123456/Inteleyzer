# How to Test ADR Form Submission from Browser

## Step-by-Step Instructions

### 1. **Login as Pharmacy User**
   - Go to: `http://127.0.0.1:5000/pharmacy/login`
   - Use any pharmacy credentials from the database
   - Example: 
     - Email: `cvs.downtown@pharmacy.com` (or similar)
     - You can check available pharmacy users in your database

### 2. **Navigate to ADR Report Form**
   - After login, go to: `http://127.0.0.1:5000/pharmacy/report`
   - OR click "Report ADR" from the pharmacy dashboard

### 3. **Fill the Form**
   - **Patient Information Section**:
     - Name: Any name
     - Age: Any number
     - Gender: Select M/F
     - Weight: Any number
     - Phone: Any phone number (e.g., +918595687463)
     - Email: Any valid email (e.g., test@example.com)
   
   - **Drug Information Section**:
     - Drug Name: Any drug
     - Dosage: e.g., 500mg
     - Route: oral/iv/etc
     - Start Date: Any date
     - Reaction Date: Any date
   
   - **Reaction Details Section**:
     - Reaction Category: Select from list
     - Severity: mild/moderate/severe
     - Outcome: recovered/improving/etc
   
   - **Medical History**:
     - Concomitant Medications: Optional
     - Medical History: Optional
     - Additional Notes: Optional

### 4. **Submit the Form**
   - Click the blue "✉️ Submit Report" button
   - **Wait 2-3 seconds** for the request to process

### 5. **See the Success Modal**
   - A white dialog box should appear (centered on screen)
   - It will show:
     - ✅ Submission ID
     - ✅ Patients Created: 1
     - ✅ Email Status: **✅ Sent**
     - ✅ WhatsApp Status: **✅ Sent**
     - 📅 Follow-up Cycle: Day 1 → Day 3 → Day 5 → Day 7

### 6. **Verify Messages Sent**
   - Check your email inbox for follow-up email
   - If WhatsApp phone is verified in Twilio sandbox, check WhatsApp for language selection menu

---

## If Nothing Happens When Clicking Submit

### Check 1: Browser Console
1. Press `F12` to open Developer Tools
2. Go to **Console** tab
3. Try submitting again
4. Look for any red error messages
5. **Send screenshot of any errors**

### Check 2: Network Tab
1. Open Developer Tools (`F12`)
2. Go to **Network** tab
3. Try submitting the form
4. Look for a request to `/api/pharmacy/reports/submit`
5. Check if it returns Status 201 (success) or error
6. **Send screenshot of the request/response**

### Check 3: Authentication
- Make sure you're **logged in** as pharmacy user
- Check session cookie in browser (F12 → Application → Cookies)
- Should have `session` cookie with user data

### Check 4: Form Validation
- Make sure **all required fields** are filled
- Required fields are marked with red asterisk (*)
- Verify no fields are empty

---

## Server-Side Testing

If you want to verify from command line without browser:

```bash
# Run the test script
python test_adr_submit.py
```

Expected output:
```
[SUCCESS] Report submitted!
- Submission ID: SUB-...
- Patients Created: 1
- Email Sent: True
- WhatsApp Sent: True
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Form button doesn't respond | Check if you're logged in as pharmacy user |
| Form submission fails with 401 | Login required - go to /pharmacy/login first |
| Modal doesn't appear | Check browser console (F12) for JavaScript errors |
| Email/WhatsApp show "False" | Check API response status (should be 201, not 200) |
| Page redirects to login | Session expired - login again |

---

## Troubleshooting Script

If submissions are still failing, run this to diagnose:

```bash
# Check if pharmacy users exist
python -c "from models import User, db; from app import app; 
app.app_context().push(); 
users = User.query.filter_by(role='pharmacy').all(); 
print(f'Found {len(users)} pharmacy users')"

# Check if Flask server is running
netstat -ano | grep 5000

# Check database connection
python -c "from app import app, db; 
app.app_context().push(); 
print(f'Database OK')"
```

---

## Still Having Issues?

Please provide:
1. Screenshot from browser console (F12)
2. Screenshot from Network tab showing `/api/pharmacy/reports/submit` request
3. Response status code and body
4. Is the form visible on screen? (can you see the input fields?)
5. Do you see "Submitting..." text appear when clicking submit?
