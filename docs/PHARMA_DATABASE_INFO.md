# Pharmaceutical Database - Complete Information

## 📊 Database Summary

The database has been populated with comprehensive mock data including:

- **8 Pharmaceutical Companies** with login credentials
- **27 Drugs** across all companies (drug portfolio)
- **150 Patient Reports** (adverse event reports)
  - 90 Low Risk (60%)
  - 45 Medium Risk (30%)
  - 15 High Risk (10%)
- **49 Safety Alerts** from various companies
  - 2 Critical
  - 10 High
  - 17 Medium
  - 20 Low

---

## 🔐 Company Login Credentials

Use these credentials to login to any pharmaceutical company dashboard:

### 1. Novartis Pharmaceuticals
- **Email:** `admin@novartis.com`
- **Password:** `novartis2024`
- **Drugs:** Diovan, Gilenya, Cosentyx, Entresto

### 2. Pfizer Inc.
- **Email:** `admin@pfizer.com`
- **Password:** `pfizer2024`
- **Drugs:** Lipitor, Eliquis, Xeljanz, Prevnar 13

### 3. Johnson & Johnson
- **Email:** `admin@jnj.com`
- **Password:** `jnj2024`
- **Drugs:** Stelara, Xarelto, Invega Sustenna, Darzalex

### 4. Roche Pharmaceuticals
- **Email:** `admin@roche.com`
- **Password:** `roche2024`
- **Drugs:** Avastin, Herceptin, Rituxan

### 5. AstraZeneca
- **Email:** `admin@astrazeneca.com`
- **Password:** `astra2024`
- **Drugs:** Farxiga, Symbicort, Tagrisso

### 6. Merck & Co.
- **Email:** `admin@merck.com`
- **Password:** `merck2024`
- **Drugs:** Januvia, Keytruda, Gardasil 9

### 7. GSK (GlaxoSmithKline)
- **Email:** `admin@gsk.com`
- **Password:** `gsk2024`
- **Drugs:** Advair, Shingrix, Nucala

### 8. Sanofi
- **Email:** `admin@sanofi.com`
- **Password:** `sanofi2024`
- **Drugs:** Lantus, Dupixent, Plavix

---

## 📁 Excel File Export

All database information has been exported to:
**`pharma_complete_database.xlsx`**

### Excel Sheets:

1. **Pharma Companies** - All company details with credentials
2. **Drug Portfolio** - Complete list of all drugs with descriptions, active ingredients, and AI risk assessments
3. **Patient Reports** - All adverse event reports with patient demographics and risk levels
4. **Safety Alerts** - All safety alerts issued by companies
5. **Statistics** - Summary statistics and metrics

---

## 🌐 How to Access

1. **Start the Server:**
   ```bash
   cd C:\Users\SONUR\projects\Novartis\backend
   python app.py
   ```

2. **Open Browser:**
   Navigate to `http://127.0.0.1:5000`

3. **Login:**
   - Click "Login"
   - Use any of the email/password combinations above
   - You'll be redirected to the pharma dashboard

---

## 📊 What You Can See in Each Dashboard

### Pharma Dashboard (`/pharma/dashboard`)
- Total number of adverse event reports
- High-risk patient count
- Average patient age
- Gender distribution charts
- Risk level distribution
- Recent patient reports with symptoms
- Statistics and analytics

### Drug Portfolio (`/pharma/drugs`)
- Complete list of all your company's drugs
- Drug descriptions and active ingredients
- AI-powered risk assessments (Low, Medium, High)
- Risk analysis details
- Ability to add new drugs

### Reports (`/pharma/reports`)
- All adverse event reports from patients
- Patient demographics (name, age, gender)
- Drug being taken
- Reported symptoms
- Risk level classification
- Contact information

### Alerts (via API)
- Safety alerts sent by your company
- Alert severity levels
- Drug names and messages
- Alert status (read/unread)

---

## 🔬 Drug Portfolio Breakdown

### By Risk Level:
- **High Risk (8 drugs):** Eliquis, Xeljanz, Xarelto, Avastin, Tagrisso, Keytruda
- **Medium Risk (11 drugs):** Gilenya, Cosentyx, Entresto, Stelara, Invega Sustenna, Darzalex, Herceptin, Rituxan, Farxiga, Lantus, Plavix
- **Low Risk (8 drugs):** Diovan, Lipitor, Prevnar 13, Symbicort, Januvia, Gardasil 9, Advair, Shingrix, Nucala, Dupixent

### Example Drugs with Details:

**High Risk Example - Eliquis (Pfizer):**
- Factor Xa inhibitor anticoagulant
- Apixaban 2.5mg, 5mg
- AI Assessment: High - Risk of bleeding. Requires careful dosing in renal impairment.

**Medium Risk Example - Gilenya (Novartis):**
- Sphingosine 1-phosphate receptor modulator for MS
- Fingolimod 0.5mg
- AI Assessment: Medium - Requires cardiac monitoring at initiation.

**Low Risk Example - Lipitor (Pfizer):**
- HMG-CoA reductase inhibitor for cholesterol
- Atorvastatin Calcium 10mg, 20mg, 40mg, 80mg
- AI Assessment: Low - Well-established safety profile.

---

## 📈 Patient Reports Sample Data

The system contains 150 realistic patient reports with:
- Diverse patient names and demographics
- Age range: 25-85 years
- Gender distribution: Male, Female, Other
- Phone numbers in US format
- Associated drugs from the portfolio
- Realistic symptoms based on risk level
- Created dates spanning last 6 months

### Risk Level Symptoms Examples:

**Low Risk Symptoms:**
- Mild headache
- Slight dizziness when standing
- Occasional nausea
- Minor stomach upset

**Medium Risk Symptoms:**
- Persistent headache and dizziness
- Moderate nausea with decreased appetite
- Frequent heart palpitations
- Moderate skin rash with itching

**High Risk Symptoms:**
- Severe chest pain and difficulty breathing
- Extreme dizziness with fainting episodes
- Severe allergic reaction with hives
- Uncontrolled bleeding and bruising

---

## 🚨 Safety Alerts Sample Data

49 realistic safety alerts distributed across severity levels:

**Critical (2):** Immediate action required, voluntary recalls
**High (10):** Urgent safety concerns, serious adverse reactions
**Medium (17):** Important updates, new contraindications
**Low (20):** Routine updates, minor adverse events

### Sample Alerts:

- "CRITICAL SAFETY ALERT for Eliquis: Voluntary recall initiated due to serious adverse events."
- "Important safety information for Xeljanz: Black box warning being added for hepatotoxicity risk."
- "Routine safety update for Lipitor: Minor adverse events reported in post-market surveillance."

---

## 🎯 Features Available

1. **View Dashboard Statistics**
2. **Browse Drug Portfolio**
3. **View Patient Reports**
4. **Add New Drugs** (through UI)
5. **Send Safety Alerts** (through UI)
6. **Export Data** (via Excel file)
7. **AI Risk Assessment** (automatic for new drugs)
8. **Real-time Data Updates**

---

## 📝 Notes

- All data is stored in SQLite database: `medsafe.db`
- Database is automatically created when you run `app.py`
- To regenerate data, simply run `populate_pharma_data.py` again
- Excel file is regenerated each time the script runs
- All passwords are in plain text for demo purposes (would be hashed in production)
- The system includes a default doctor account for patient assignment

---

## 🛠️ Technical Details

**Database Tables:**
- `user` - Stores doctors and pharma companies
- `patient` - Adverse event reports
- `drug` - Drug portfolio
- `alert` - Safety alerts
- `doctor_patient` - Association table for many-to-many relationships

**Technologies:**
- Flask (Python web framework)
- SQLAlchemy (ORM)
- SQLite (Database)
- Pandas + OpenPyXL (Excel export)
- Bootstrap (Frontend styling)

---

## ✅ Verification Checklist

- [x] 8 pharmaceutical companies created
- [x] 27 drugs in portfolio with detailed information
- [x] 150 patient reports with realistic data
- [x] 49 safety alerts across all severity levels
- [x] Excel file with 5 sheets of data
- [x] All companies have login credentials
- [x] Data viewable through pharma dashboards
- [x] AI risk assessments for all drugs
- [x] Realistic patient demographics and symptoms

---

**Last Updated:** January 23, 2026
**File Location:** `C:\Users\SONUR\projects\Novartis\backend\`
**Excel File:** `pharma_complete_database.xlsx`
