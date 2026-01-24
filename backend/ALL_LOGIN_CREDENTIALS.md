# Complete Database - All Login Credentials

## 🌐 Access URL
**http://127.0.0.1:5000/login**

---

## 📊 Database Summary
- **Pharmaceutical Companies**: 8
- **Doctors**: 10
- **Local Pharmacies**: 10
- **Total Drugs**: 27
- **ADR Reports**: 200 (120 from doctors, 80 from pharmacies)
- **Safety Alerts**: 60

---

## 🏢 Pharmaceutical Companies (8)

| Company | Email | Password |
|---------|-------|----------|
| Novartis Pharmaceuticals | admin@novartis.com | novartis2024 |
| Pfizer Inc. | admin@pfizer.com | pfizer2024 |
| Johnson & Johnson | admin@jnj.com | jnj2024 |
| Roche Pharmaceuticals | admin@roche.com | roche2024 |
| AstraZeneca | admin@astrazeneca.com | astra2024 |
| Merck & Co. | admin@merck.com | merck2024 |
| GSK (GlaxoSmithKline) | admin@gsk.com | gsk2024 |
| Sanofi | admin@sanofi.com | sanofi2024 |

---

## 👨‍⚕️ Doctors (10)

| Name | Email | Password | Specialty |
|------|-------|----------|-----------|
| Dr. Emily Chen | emily.chen@hospital.com | doctor123 | Cardiology |
| Dr. Michael Rodriguez | m.rodriguez@clinic.com | doctor123 | Internal Medicine |
| Dr. Sarah Johnson | sarah.j@medcenter.com | doctor123 | Oncology |
| Dr. David Kim | d.kim@hospital.com | doctor123 | Neurology |
| Dr. Jennifer Martinez | j.martinez@clinic.com | doctor123 | Psychiatry |
| Dr. Robert Taylor | r.taylor@medcenter.com | doctor123 | Dermatology |
| Dr. Lisa Anderson | l.anderson@hospital.com | doctor123 | Rheumatology |
| Dr. James Wilson | j.wilson@clinic.com | doctor123 | Endocrinology |
| Dr. Maria Garcia | m.garcia@medcenter.com | doctor123 | Gastroenterology |
| Dr. William Brown | w.brown@hospital.com | doctor123 | Pulmonology |

---

## 💊 Local Pharmacies (10)

| Name | Email | Password | Location |
|------|-------|----------|----------|
| CVS Pharmacy - Downtown | downtown@cvs-pharmacy.com | pharmacy123 | 123 Main St, Downtown |
| Walgreens - Westside | westside@walgreens.com | pharmacy123 | 456 West Ave, Westside |
| Rite Aid - Eastgate | eastgate@riteaid.com | pharmacy123 | 789 East Blvd, Eastgate |
| Community Pharmacy | info@communitypharmacy.com | pharmacy123 | 321 Oak Street |
| HealthMart Pharmacy | contact@healthmart.com | pharmacy123 | 654 Pine Ave |
| MedPlus Pharmacy | info@medplus.com | pharmacy123 | 987 Maple Dr |
| Express Scripts Pharmacy | support@expressscripts.com | pharmacy123 | 147 Cedar Ln |
| Walmart Pharmacy | pharmacy@walmart.com | pharmacy123 | 258 Commerce St |
| Costco Pharmacy | pharmacy@costco.com | pharmacy123 | 369 Wholesale Ave |
| Target Pharmacy | pharmacy@target.com | pharmacy123 | 741 Retail Blvd |

---

## 💊 Drug Portfolio by Company

### Novartis Pharmaceuticals (4 drugs)
- **Diovan** - ARB for hypertension (Low Risk)
- **Gilenya** - MS treatment (Medium Risk)
- **Cosentyx** - IL-17A inhibitor (Medium Risk)
- **Entresto** - Heart failure treatment (Medium Risk)

### Pfizer Inc. (4 drugs)
- **Lipitor** - Statin for cholesterol (Low Risk)
- **Eliquis** - Anticoagulant (High Risk)
- **Xeljanz** - JAK inhibitor (High Risk)
- **Prevnar 13** - Pneumococcal vaccine (Low Risk)

### Johnson & Johnson (4 drugs)
- **Stelara** - IL-12/IL-23 inhibitor (Medium Risk)
- **Xarelto** - Factor Xa inhibitor (High Risk)
- **Invega Sustenna** - Antipsychotic injection (Medium Risk)
- **Darzalex** - CD38 antibody (Medium Risk)

### Roche Pharmaceuticals (3 drugs)
- **Avastin** - VEGF inhibitor (High Risk)
- **Herceptin** - HER2 therapy (Medium Risk)
- **Rituxan** - CD20 antibody (Medium Risk)

### AstraZeneca (3 drugs)
- **Farxiga** - SGLT2 inhibitor (Medium Risk)
- **Symbicort** - Combination inhaler (Low Risk)
- **Tagrisso** - EGFR inhibitor (High Risk)

### Merck & Co. (3 drugs)
- **Januvia** - DPP-4 inhibitor (Low Risk)
- **Keytruda** - PD-1 inhibitor (High Risk)
- **Gardasil 9** - HPV vaccine (Low Risk)

### GSK (3 drugs)
- **Advair** - ICS/LABA inhaler (Low Risk)
- **Shingrix** - Zoster vaccine (Low Risk)
- **Nucala** - IL-5 inhibitor (Low Risk)

### Sanofi (3 drugs)
- **Lantus** - Long-acting insulin (Medium Risk)
- **Dupixent** - IL-4/IL-13 inhibitor (Low Risk)
- **Plavix** - P2Y12 inhibitor (Medium Risk)

---

## 📈 ADR Reports Distribution

- **Total Reports**: 200
  - **From Doctors**: 120 (60%)
  - **From Pharmacies**: 80 (40%)
  
- **Risk Levels**:
  - **High Risk**: 20 reports (10%)
  - **Medium Risk**: 60 reports (30%)
  - **Low Risk**: 120 reports (60%)

---

## 🚨 Safety Alerts Distribution

- **Total Alerts**: 60
  - **Critical**: 3 alerts
  - **High**: 12 alerts
  - **Medium**: 21 alerts
  - **Low**: 24 alerts

---

## 📁 Files Generated

1. **complete_database.xlsx** - Full database export with 8 sheets:
   - Pharma Companies
   - Doctors
   - Pharmacies
   - Drug Portfolio
   - ADR Reports
   - Safety Alerts
   - Statistics
   - Login Credentials

2. **medsafe.db** - SQLite database with all populated data

---

## 🔧 How to Use

1. Make sure Flask server is running: `python app.py`
2. Open browser: http://127.0.0.1:5000
3. Click "Login"
4. Select role and use credentials above
5. Explore the dashboard!

---

## 📝 Notes

- All passwords are simple for demo purposes
- Data is randomly generated with realistic patterns
- Reports are distributed over the last 180 days
- Alerts are distributed over the last 90 days
- Each doctor and pharmacy has multiple reports
- Patient names, phones, ages are randomly generated
- Symptoms match risk levels (Low/Medium/High)
