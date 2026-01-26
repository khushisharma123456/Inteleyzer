"""
Complete Database Population Script for Novartis MedSafe Platform
Includes: Pharma Companies, Doctors, Pharmacies, Drugs, Patients, Alerts
Generates comprehensive Excel export
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from models import User, Drug, Patient, Alert
from datetime import datetime, timedelta
import random
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import os

# ============================================================================
# PHARMACEUTICAL COMPANIES DATA
# ============================================================================
PHARMA_COMPANIES = [
    {"name": "Novartis Pharmaceuticals", "email": "admin@novartis.com", "password": "novartis2024"},
    {"name": "Pfizer Inc.", "email": "admin@pfizer.com", "password": "pfizer2024"},
    {"name": "Johnson & Johnson", "email": "admin@jnj.com", "password": "jnj2024"},
    {"name": "Roche Pharmaceuticals", "email": "admin@roche.com", "password": "roche2024"},
    {"name": "AstraZeneca", "email": "admin@astrazeneca.com", "password": "astra2024"},
    {"name": "Merck & Co.", "email": "admin@merck.com", "password": "merck2024"},
    {"name": "GSK (GlaxoSmithKline)", "email": "admin@gsk.com", "password": "gsk2024"},
    {"name": "Sanofi", "email": "admin@sanofi.com", "password": "sanofi2024"}
]

# ============================================================================
# DOCTORS DATA
# ============================================================================
DOCTORS = [
    {"name": "Dr. Emily Chen", "email": "emily.chen@hospital.com", "password": "doctor123", "specialty": "Cardiology"},
    {"name": "Dr. Michael Rodriguez", "email": "m.rodriguez@clinic.com", "password": "doctor123", "specialty": "Internal Medicine"},
    {"name": "Dr. Sarah Johnson", "email": "sarah.j@medcenter.com", "password": "doctor123", "specialty": "Oncology"},
    {"name": "Dr. David Kim", "email": "d.kim@hospital.com", "password": "doctor123", "specialty": "Neurology"},
    {"name": "Dr. Jennifer Martinez", "email": "j.martinez@clinic.com", "password": "doctor123", "specialty": "Psychiatry"},
    {"name": "Dr. Robert Taylor", "email": "r.taylor@medcenter.com", "password": "doctor123", "specialty": "Dermatology"},
    {"name": "Dr. Lisa Anderson", "email": "l.anderson@hospital.com", "password": "doctor123", "specialty": "Rheumatology"},
    {"name": "Dr. James Wilson", "email": "j.wilson@clinic.com", "password": "doctor123", "specialty": "Endocrinology"},
    {"name": "Dr. Maria Garcia", "email": "m.garcia@medcenter.com", "password": "doctor123", "specialty": "Gastroenterology"},
    {"name": "Dr. William Brown", "email": "w.brown@hospital.com", "password": "doctor123", "specialty": "Pulmonology"}
]

# ============================================================================
# LOCAL PHARMACIES DATA
# ============================================================================
PHARMACIES = [
    {"name": "CVS Pharmacy - Downtown", "email": "downtown@cvs-pharmacy.com", "password": "pharmacy123", "location": "123 Main St, Downtown"},
    {"name": "Walgreens - Westside", "email": "westside@walgreens.com", "password": "pharmacy123", "location": "456 West Ave, Westside"},
    {"name": "Rite Aid - Eastgate", "email": "eastgate@riteaid.com", "password": "pharmacy123", "location": "789 East Blvd, Eastgate"},
    {"name": "Community Pharmacy", "email": "info@communitypharmacy.com", "password": "pharmacy123", "location": "321 Oak Street"},
    {"name": "HealthMart Pharmacy", "email": "contact@healthmart.com", "password": "pharmacy123", "location": "654 Pine Ave"},
    {"name": "MedPlus Pharmacy", "email": "info@medplus.com", "password": "pharmacy123", "location": "987 Maple Dr"},
    {"name": "Express Scripts Pharmacy", "email": "support@expressscripts.com", "password": "pharmacy123", "location": "147 Cedar Ln"},
    {"name": "Walmart Pharmacy", "email": "pharmacy@walmart.com", "password": "pharmacy123", "location": "258 Commerce St"},
    {"name": "Costco Pharmacy", "email": "pharmacy@costco.com", "password": "pharmacy123", "location": "369 Wholesale Ave"},
    {"name": "Target Pharmacy", "email": "pharmacy@target.com", "password": "pharmacy123", "location": "741 Retail Blvd"}
]

# ============================================================================
# DRUG PORTFOLIO DATA
# ============================================================================
DRUG_DATA = {
    "Novartis Pharmaceuticals": [
        {"name": "Diovan", "description": "ARB for hypertension", "active_ingredients": "Valsartan 80-320mg", "ai_risk": "Low", "ai_details": "Well-established safety profile"},
        {"name": "Gilenya", "description": "MS treatment", "active_ingredients": "Fingolimod 0.5mg", "ai_risk": "Medium", "ai_details": "Cardiac monitoring required"},
        {"name": "Cosentyx", "description": "IL-17A inhibitor", "active_ingredients": "Secukinumab 150-300mg", "ai_risk": "Medium", "ai_details": "Monitor for infections"},
        {"name": "Entresto", "description": "Heart failure treatment", "active_ingredients": "Sacubitril/Valsartan", "ai_risk": "Medium", "ai_details": "Monitor blood pressure"}
    ],
    "Pfizer Inc.": [
        {"name": "Lipitor", "description": "Statin for cholesterol", "active_ingredients": "Atorvastatin 10-80mg", "ai_risk": "Low", "ai_details": "Monitor liver enzymes"},
        {"name": "Eliquis", "description": "Anticoagulant", "active_ingredients": "Apixaban 2.5-5mg", "ai_risk": "High", "ai_details": "Bleeding risk monitoring essential"},
        {"name": "Xeljanz", "description": "JAK inhibitor", "active_ingredients": "Tofacitinib 5-10mg", "ai_risk": "High", "ai_details": "Black box warning for infections"},
        {"name": "Prevnar 13", "description": "Pneumococcal vaccine", "active_ingredients": "13-valent conjugate", "ai_risk": "Low", "ai_details": "Standard vaccine safety"}
    ],
    "Johnson & Johnson": [
        {"name": "Stelara", "description": "IL-12/IL-23 inhibitor", "active_ingredients": "Ustekinumab 45-90mg", "ai_risk": "Medium", "ai_details": "Monitor for infections"},
        {"name": "Xarelto", "description": "Factor Xa inhibitor", "active_ingredients": "Rivaroxaban 10-20mg", "ai_risk": "High", "ai_details": "Bleeding risk assessment"},
        {"name": "Invega Sustenna", "description": "Antipsychotic injection", "active_ingredients": "Paliperidone", "ai_risk": "Medium", "ai_details": "Monitor metabolic changes"},
        {"name": "Darzalex", "description": "CD38 antibody", "active_ingredients": "Daratumumab", "ai_risk": "Medium", "ai_details": "Infusion reactions common"}
    ],
    "Roche Pharmaceuticals": [
        {"name": "Avastin", "description": "VEGF inhibitor", "active_ingredients": "Bevacizumab", "ai_risk": "High", "ai_details": "Black box warnings multiple"},
        {"name": "Herceptin", "description": "HER2 therapy", "active_ingredients": "Trastuzumab", "ai_risk": "Medium", "ai_details": "Cardiac monitoring essential"},
        {"name": "Rituxan", "description": "CD20 antibody", "active_ingredients": "Rituximab", "ai_risk": "Medium", "ai_details": "Monitor for PML"}
    ],
    "AstraZeneca": [
        {"name": "Farxiga", "description": "SGLT2 inhibitor", "active_ingredients": "Dapagliflozin 5-10mg", "ai_risk": "Medium", "ai_details": "DKA risk monitoring"},
        {"name": "Symbicort", "description": "Combination inhaler", "active_ingredients": "Budesonide/Formoterol", "ai_risk": "Low", "ai_details": "Standard ICS/LABA risks"},
        {"name": "Tagrisso", "description": "EGFR inhibitor", "active_ingredients": "Osimertinib", "ai_risk": "High", "ai_details": "ILD and cardiac monitoring"}
    ],
    "Merck & Co.": [
        {"name": "Januvia", "description": "DPP-4 inhibitor", "active_ingredients": "Sitagliptin 25-100mg", "ai_risk": "Low", "ai_details": "Monitor for pancreatitis"},
        {"name": "Keytruda", "description": "PD-1 inhibitor", "active_ingredients": "Pembrolizumab", "ai_risk": "High", "ai_details": "Immune-related AEs monitoring"},
        {"name": "Gardasil 9", "description": "HPV vaccine", "active_ingredients": "9-valent HPV vaccine", "ai_risk": "Low", "ai_details": "Standard vaccine profile"}
    ],
    "GSK (GlaxoSmithKline)": [
        {"name": "Advair", "description": "ICS/LABA inhaler", "active_ingredients": "Fluticasone/Salmeterol", "ai_risk": "Low", "ai_details": "Monitor for pneumonia in COPD"},
        {"name": "Shingrix", "description": "Zoster vaccine", "active_ingredients": "Recombinant glycoprotein E", "ai_risk": "Low", "ai_details": "Common reactogenicity"},
        {"name": "Nucala", "description": "IL-5 inhibitor", "active_ingredients": "Mepolizumab", "ai_risk": "Low", "ai_details": "Generally well tolerated"}
    ],
    "Sanofi": [
        {"name": "Lantus", "description": "Long-acting insulin", "active_ingredients": "Insulin glargine 100U/mL", "ai_risk": "Medium", "ai_details": "Hypoglycemia risk management"},
        {"name": "Dupixent", "description": "IL-4/IL-13 inhibitor", "active_ingredients": "Dupilumab", "ai_risk": "Low", "ai_details": "Monitor for conjunctivitis"},
        {"name": "Plavix", "description": "P2Y12 inhibitor", "active_ingredients": "Clopidogrel 75mg", "ai_risk": "Medium", "ai_details": "Bleeding risk assessment"}
    ]
}

# ============================================================================
# PATIENT NAMES AND SYMPTOMS
# ============================================================================
FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
               "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
               "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
               "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
               "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
               "Kevin", "Dorothy", "Brian", "Carol", "George", "Amanda", "Edward", "Melissa",
               "Ronald", "Deborah", "Timothy", "Stephanie", "Jason", "Rebecca", "Jeffrey", "Sharon"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
              "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
              "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams"]

SYMPTOMS_LOW = [
    "Mild headache", "Slight dizziness when standing", "Occasional nausea",
    "Mild fatigue", "Dry mouth", "Minor stomach upset", "Slight insomnia",
    "Mild constipation", "Minor skin irritation", "Mild drowsiness"
]

SYMPTOMS_MEDIUM = [
    "Persistent headache and dizziness", "Moderate nausea with decreased appetite",
    "Fatigue affecting daily activities", "Frequent heart palpitations",
    "Unexplained bruising", "Persistent dry cough", "Muscle weakness and joint pain",
    "Moderate skin rash with itching", "Sleep disturbances", "Moderate anxiety"
]

SYMPTOMS_HIGH = [
    "Severe chest pain and difficulty breathing", "Extreme dizziness with fainting episodes",
    "Severe allergic reaction with hives and swelling", "Irregular heartbeat with chest tightness",
    "Uncontrolled bleeding and bruising", "Severe abdominal pain with vomiting",
    "Confusion and memory problems", "Severe muscle weakness affecting mobility",
    "High fever with severe headache", "Anaphylactic symptoms"
]

# ============================================================================
# ALERT TEMPLATES
# ============================================================================
ALERT_TEMPLATES = {
    "Low": [
        "Routine safety update for {drug}: Minor adverse events reported in post-market surveillance.",
        "Information update for {drug}: New drug interaction identified with common antacids.",
        "Quality notification for {drug}: Lot recall due to packaging defect (no safety impact)."
    ],
    "Medium": [
        "Safety alert for {drug}: Increased incidence of mild allergic reactions reported.",
        "Important update for {drug}: New contraindication identified for severe renal impairment.",
        "Pharmacovigilance notice for {drug}: Enhanced monitoring recommended for elderly patients."
    ],
    "High": [
        "URGENT: Safety concern for {drug}: Multiple reports of severe adverse reactions.",
        "Critical alert for {drug}: Potential risk of serious cardiovascular events.",
        "Important safety information for {drug}: Black box warning being added."
    ],
    "Critical": [
        "CRITICAL SAFETY ALERT for {drug}: Voluntary recall initiated due to serious adverse events.",
        "IMMEDIATE ACTION REQUIRED for {drug}: Suspension of sales pending investigation.",
        "EMERGENCY NOTIFICATION for {drug}: Life-threatening reactions reported."
    ]
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def clear_database():
    """Clear existing data"""
    print("\n=== Clearing Existing Database ===")
    try:
        db.session.query(Alert).delete()
        db.session.query(Patient).delete()
        db.session.query(Drug).delete()
        db.session.query(User).delete()
        db.session.commit()
        print("✓ Database cleared successfully")
    except Exception as e:
        print(f"✗ Error clearing database: {e}")
        db.session.rollback()

def create_pharma_companies():
    """Create pharmaceutical companies"""
    print("\n=== Creating Pharmaceutical Companies ===")
    companies = []
    for data in PHARMA_COMPANIES:
        existing = User.query.filter_by(email=data['email']).first()
        if not existing:
            company = User(name=data['name'], email=data['email'], 
                          password=data['password'], role='pharma')
            db.session.add(company)
            companies.append(company)
            print(f"✓ Created: {data['name']}")
        else:
            companies.append(existing)
            print(f"  Already exists: {data['name']}")
    db.session.commit()
    return companies

def create_doctors():
    """Create doctor accounts"""
    print("\n=== Creating Doctor Accounts ===")
    doctors = []
    for data in DOCTORS:
        existing = User.query.filter_by(email=data['email']).first()
        if not existing:
            doctor = User(name=data['name'], email=data['email'],
                         password=data['password'], role='doctor')
            db.session.add(doctor)
            doctors.append(doctor)
            print(f"✓ Created: {data['name']} - {data['specialty']}")
        else:
            doctors.append(existing)
            print(f"  Already exists: {data['name']}")
    db.session.commit()
    return doctors

def create_pharmacies():
    """Create pharmacy accounts"""
    print("\n=== Creating Local Pharmacy Accounts ===")
    pharmacies = []
    for data in PHARMACIES:
        existing = User.query.filter_by(email=data['email']).first()
        if not existing:
            pharmacy = User(name=data['name'], email=data['email'],
                           password=data['password'], role='pharmacy')
            db.session.add(pharmacy)
            pharmacies.append(pharmacy)
            print(f"✓ Created: {data['name']} - {data['location']}")
        else:
            pharmacies.append(existing)
            print(f"  Already exists: {data['name']}")
    db.session.commit()
    return pharmacies

def create_drugs(companies):
    """Create drug portfolios"""
    print("\n=== Creating Drug Portfolios ===")
    drugs = []
    for company in companies:
        if company.name in DRUG_DATA:
            print(f"\n{company.name}:")
            for drug_data in DRUG_DATA[company.name]:
                existing = Drug.query.filter_by(name=drug_data['name'], 
                                               company_id=company.id).first()
                if not existing:
                    drug = Drug(
                        name=drug_data['name'],
                        company_id=company.id,
                        description=drug_data['description'],
                        active_ingredients=drug_data['active_ingredients'],
                        ai_risk_assessment=drug_data['ai_risk'],
                        ai_risk_details=drug_data['ai_details']
                    )
                    db.session.add(drug)
                    drugs.append(drug)
                    print(f"  ✓ {drug_data['name']} (Risk: {drug_data['ai_risk']})")
                else:
                    drugs.append(existing)
    db.session.commit()
    return drugs

def create_patients(doctors, pharmacies, drugs, num_patients=200):
    """Create patient/ADR reports"""
    print(f"\n=== Creating {num_patients} Patient/ADR Reports ===")
    patients = []
    
    # Split reports between doctors and pharmacies
    doctor_reports = int(num_patients * 0.6)  # 60% from doctors
    pharmacy_reports = num_patients - doctor_reports  # 40% from pharmacies
    
    risk_distribution = {
        'Low': int(num_patients * 0.60),
        'Medium': int(num_patients * 0.30),
        'High': int(num_patients * 0.10)
    }
    
    count = 0
    for risk_level, total_count in risk_distribution.items():
        for i in range(total_count):
            count += 1
            
            # Decide if from doctor or pharmacy
            if count <= doctor_reports:
                creator = random.choice(doctors)
                id_prefix = 'DOC'
            else:
                creator = random.choice(pharmacies)
                id_prefix = 'PH'
            
            # Generate patient data
            first_name = random.choice(FIRST_NAMES)
            last_name = random.choice(LAST_NAMES)
            name = f"{first_name} {last_name}"
            
            # Select symptoms
            if risk_level == 'Low':
                symptoms = random.choice(SYMPTOMS_LOW)
            elif risk_level == 'Medium':
                symptoms = random.choice(SYMPTOMS_MEDIUM)
            else:
                symptoms = random.choice(SYMPTOMS_HIGH)
            
            drug = random.choice(drugs)
            patient_id = f"{id_prefix}-{random.randint(1000, 9999)}"
            
            # Check uniqueness
            while Patient.query.get(patient_id):
                patient_id = f"{id_prefix}-{random.randint(1000, 9999)}"
            
            patient = Patient(
                id=patient_id,
                created_by=creator.id,
                name=name,
                phone=f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
                age=random.randint(25, 85),
                gender=random.choice(['Male', 'Female', 'Male', 'Female', 'Other']),
                drug_name=drug.name,
                symptoms=symptoms,
                risk_level=risk_level,
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 180))
            )
            
            # Link to doctors for collaboration
            if creator.role == 'doctor':
                patient.doctors.append(creator)
            else:
                # Pharmacy reports might also be linked to a doctor
                if random.random() > 0.5:
                    patient.doctors.append(random.choice(doctors))
            
            db.session.add(patient)
            patients.append(patient)
            
            if count % 50 == 0:
                print(f"  ✓ Created {count}/{num_patients} reports...")
    
    db.session.commit()
    print(f"✓ Total reports created: {len(patients)}")
    return patients

def create_alerts(companies, drugs, num_alerts=60):
    """Create safety alerts"""
    print(f"\n=== Creating {num_alerts} Safety Alerts ===")
    alerts = []
    
    severity_distribution = {
        'Low': int(num_alerts * 0.40),
        'Medium': int(num_alerts * 0.35),
        'High': int(num_alerts * 0.20),
        'Critical': int(num_alerts * 0.05)
    }
    
    for severity, count in severity_distribution.items():
        for i in range(count):
            company = random.choice(companies)
            company_drugs = [d for d in drugs if d.company_id == company.id]
            if not company_drugs:
                continue
            
            drug = random.choice(company_drugs)
            template = random.choice(ALERT_TEMPLATES[severity])
            message = template.format(drug=drug.name)
            
            alert = Alert(
                drug_name=drug.name,
                message=message,
                severity=severity,
                sender_id=company.id,
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 90)),
                is_read=random.choice([True, False, False])
            )
            
            db.session.add(alert)
            alerts.append(alert)
    
    db.session.commit()
    print(f"✓ Total alerts created: {len(alerts)}")
    return alerts

def export_to_excel(companies, doctors, pharmacies, drugs, patients, alerts):
    """Export all data to Excel"""
    print("\n=== Exporting Data to Excel ===")
    
    excel_file = 'C:\\Users\\SONUR\\projects\\Novartis\\docs\\complete_database.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Sheet 1: Pharma Companies
        companies_data = [{
            'ID': c.id,
            'Name': c.name,
            'Email': c.email,
            'Password': c.password,
            'Role': c.role,
            'Drugs Count': len([d for d in drugs if d.company_id == c.id])
        } for c in companies]
        pd.DataFrame(companies_data).to_excel(writer, sheet_name='Pharma Companies', index=False)
        print("✓ Exported: Pharma Companies")
        
        # Sheet 2: Doctors
        doctors_data = [{
            'ID': d.id,
            'Name': d.name,
            'Email': d.email,
            'Password': d.password,
            'Role': d.role,
            'Patients Assigned': len([p for p in patients if d in p.doctors])
        } for d in doctors]
        pd.DataFrame(doctors_data).to_excel(writer, sheet_name='Doctors', index=False)
        print("✓ Exported: Doctors")
        
        # Sheet 3: Pharmacies
        pharmacies_data = [{
            'ID': p.id,
            'Name': p.name,
            'Email': p.email,
            'Password': p.password,
            'Role': p.role,
            'Reports Filed': len([r for r in patients if r.created_by == p.id])
        } for p in pharmacies]
        pd.DataFrame(pharmacies_data).to_excel(writer, sheet_name='Pharmacies', index=False)
        print("✓ Exported: Pharmacies")
        
        # Sheet 4: Drugs
        drugs_data = [{
            'ID': d.id,
            'Drug Name': d.name,
            'Company': d.company.name,
            'Description': d.description,
            'Active Ingredients': d.active_ingredients,
            'AI Risk': d.ai_risk_assessment,
            'Risk Details': d.ai_risk_details,
            'Created Date': d.created_at.strftime('%Y-%m-%d')
        } for d in drugs]
        pd.DataFrame(drugs_data).to_excel(writer, sheet_name='Drug Portfolio', index=False)
        print("✓ Exported: Drug Portfolio")
        
        # Sheet 5: Patients/ADR Reports
        patients_data = [{
            'Report ID': p.id,
            'Patient Name': p.name,
            'Phone': p.phone,
            'Age': p.age,
            'Gender': p.gender,
            'Drug': p.drug_name,
            'Symptoms': p.symptoms,
            'Risk Level': p.risk_level,
            'Reported By': User.query.get(p.created_by).name if p.created_by else 'N/A',
            'Reporter Role': User.query.get(p.created_by).role if p.created_by else 'N/A',
            'Date': p.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for p in patients]
        pd.DataFrame(patients_data).to_excel(writer, sheet_name='ADR Reports', index=False)
        print("✓ Exported: ADR Reports")
        
        # Sheet 6: Alerts
        alerts_data = [{
            'ID': a.id,
            'Drug': a.drug_name,
            'Message': a.message,
            'Severity': a.severity,
            'Company': a.sender.name,
            'Date': a.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'Read': 'Yes' if a.is_read else 'No'
        } for a in alerts]
        pd.DataFrame(alerts_data).to_excel(writer, sheet_name='Safety Alerts', index=False)
        print("✓ Exported: Safety Alerts")
        
        # Sheet 7: Statistics
        stats = [{
            'Metric': 'Total Pharma Companies', 'Value': len(companies)
        }, {
            'Metric': 'Total Doctors', 'Value': len(doctors)
        }, {
            'Metric': 'Total Pharmacies', 'Value': len(pharmacies)
        }, {
            'Metric': 'Total Drugs', 'Value': len(drugs)
        }, {
            'Metric': 'Total ADR Reports', 'Value': len(patients)
        }, {
            'Metric': 'Reports from Doctors', 'Value': len([p for p in patients if p.id.startswith('DOC')])
        }, {
            'Metric': 'Reports from Pharmacies', 'Value': len([p for p in patients if p.id.startswith('PH')])
        }, {
            'Metric': 'High Risk Reports', 'Value': len([p for p in patients if p.risk_level == 'High'])
        }, {
            'Metric': 'Medium Risk Reports', 'Value': len([p for p in patients if p.risk_level == 'Medium'])
        }, {
            'Metric': 'Low Risk Reports', 'Value': len([p for p in patients if p.risk_level == 'Low'])
        }, {
            'Metric': 'Total Safety Alerts', 'Value': len(alerts)
        }, {
            'Metric': 'Critical Alerts', 'Value': len([a for a in alerts if a.severity == 'Critical'])
        }, {
            'Metric': 'High Severity Alerts', 'Value': len([a for a in alerts if a.severity == 'High'])
        }, {
            'Metric': 'Unread Alerts', 'Value': len([a for a in alerts if not a.is_read])
        }]
        pd.DataFrame(stats).to_excel(writer, sheet_name='Statistics', index=False)
        print("✓ Exported: Statistics")
        
        # Sheet 8: Login Credentials
        all_users = companies + doctors + pharmacies
        credentials = [{
            'Role': u.role.upper(),
            'Name': u.name,
            'Email': u.email,
            'Password': u.password,
            'Login URL': f'http://127.0.0.1:5000/login'
        } for u in all_users]
        pd.DataFrame(credentials).to_excel(writer, sheet_name='Login Credentials', index=False)
        print("✓ Exported: Login Credentials")
    
    print(f"\n✅ Excel file created: {excel_file}")
    return excel_file

def print_summary(companies, doctors, pharmacies, drugs, patients, alerts):
    """Print summary"""
    print("\n" + "="*80)
    print("DATABASE POPULATION COMPLETE")
    print("="*80)
    
    print("\n📊 SUMMARY:")
    print(f"  • Pharmaceutical Companies: {len(companies)}")
    print(f"  • Doctors: {len(doctors)}")
    print(f"  • Local Pharmacies: {len(pharmacies)}")
    print(f"  • Total Drugs: {len(drugs)}")
    print(f"  • ADR Reports: {len(patients)}")
    print(f"    - From Doctors: {len([p for p in patients if p.id.startswith('DOC')])}")
    print(f"    - From Pharmacies: {len([p for p in patients if p.id.startswith('PH')])}")
    print(f"    - High Risk: {len([p for p in patients if p.risk_level == 'High'])}")
    print(f"    - Medium Risk: {len([p for p in patients if p.risk_level == 'Medium'])}")
    print(f"    - Low Risk: {len([p for p in patients if p.risk_level == 'Low'])}")
    print(f"  • Safety Alerts: {len(alerts)}")
    print(f"    - Critical: {len([a for a in alerts if a.severity == 'Critical'])}")
    print(f"    - High: {len([a for a in alerts if a.severity == 'High'])}")
    print(f"    - Medium: {len([a for a in alerts if a.severity == 'Medium'])}")
    print(f"    - Low: {len([a for a in alerts if a.severity == 'Low'])}")
    
    print("\n🔐 SAMPLE LOGIN CREDENTIALS:")
    print("-" * 80)
    print("\nPharma Company:")
    print(f"  Email: admin@novartis.com | Password: novartis2024")
    print("\nDoctor:")
    print(f"  Email: emily.chen@hospital.com | Password: doctor123")
    print("\nPharmacy:")
    print(f"  Email: downtown@cvs-pharmacy.com | Password: pharmacy123")
    
    print("\n" + "="*80)

def main():
    """Main execution"""
    print("="*80)
    print("COMPLETE DATABASE POPULATION SCRIPT")
    print("="*80)
    
    with app.app_context():
        clear_database()
        
        companies = create_pharma_companies()
        doctors = create_doctors()
        pharmacies = create_pharmacies()
        drugs = create_drugs(companies)
        patients = create_patients(doctors, pharmacies, drugs, num_patients=200)
        alerts = create_alerts(companies, drugs, num_alerts=60)
        
        excel_file = export_to_excel(companies, doctors, pharmacies, drugs, patients, alerts)
        print_summary(companies, doctors, pharmacies, drugs, patients, alerts)
        
        print(f"\n✅ All data populated successfully!")
        print(f"📁 Excel file: {excel_file}")
        print(f"🌐 Access at: http://127.0.0.1:5000")

if __name__ == '__main__':
    main()
