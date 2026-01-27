"""
Script to populate the database with comprehensive pharmaceutical company mock data
Includes: Companies, Drugs, Patients (Reports), Alerts
Also generates Excel files for all data
"""

from app import app, db
from models import User, Drug, Patient, Alert
from datetime import datetime, timedelta
import random
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import os

# Pharmaceutical Companies Data
PHARMA_COMPANIES = [
    {
        "name": "Novartis Pharmaceuticals",
        "email": "admin@novartis.com",
        "password": "novartis2024"
    },
    {
        "name": "Pfizer Inc.",
        "email": "admin@pfizer.com",
        "password": "pfizer2024"
    },
    {
        "name": "Johnson & Johnson",
        "email": "admin@jnj.com",
        "password": "jnj2024"
    },
    {
        "name": "Roche Pharmaceuticals",
        "email": "admin@roche.com",
        "password": "roche2024"
    },
    {
        "name": "AstraZeneca",
        "email": "admin@astrazeneca.com",
        "password": "astra2024"
    },
    {
        "name": "Merck & Co.",
        "email": "admin@merck.com",
        "password": "merck2024"
    },
    {
        "name": "GSK (GlaxoSmithKline)",
        "email": "admin@gsk.com",
        "password": "gsk2024"
    },
    {
        "name": "Sanofi",
        "email": "admin@sanofi.com",
        "password": "sanofi2024"
    }
]

# Drug Portfolio Data
DRUG_DATA = {
    "Novartis Pharmaceuticals": [
        {
            "name": "Diovan",
            "description": "Angiotensin II receptor blocker for treating hypertension and heart failure",
            "active_ingredients": "Valsartan 80mg, 160mg, 320mg",
            "ai_risk": "Low",
            "ai_details": "Long-term safety profile established. Monitor for hypotension and renal function."
        },
        {
            "name": "Gilenya",
            "description": "Sphingosine 1-phosphate receptor modulator for multiple sclerosis",
            "active_ingredients": "Fingolimod 0.5mg",
            "ai_risk": "Medium",
            "ai_details": "Requires cardiac monitoring at initiation. Risk of bradycardia and infections."
        },
        {
            "name": "Cosentyx",
            "description": "IL-17A inhibitor for plaque psoriasis and psoriatic arthritis",
            "active_ingredients": "Secukinumab 150mg, 300mg",
            "ai_risk": "Medium",
            "ai_details": "Monitor for infections and inflammatory bowel disease. Generally well tolerated."
        },
        {
            "name": "Entresto",
            "description": "Neprilysin inhibitor and ARB for heart failure",
            "active_ingredients": "Sacubitril 24mg/Valsartan 26mg to 97mg/103mg",
            "ai_risk": "Medium",
            "ai_details": "Black box warning for fetal toxicity. Monitor blood pressure and renal function."
        }
    ],
    "Pfizer Inc.": [
        {
            "name": "Lipitor",
            "description": "HMG-CoA reductase inhibitor for cholesterol management",
            "active_ingredients": "Atorvastatin Calcium 10mg, 20mg, 40mg, 80mg",
            "ai_risk": "Low",
            "ai_details": "Well-established safety profile. Monitor liver enzymes and muscle pain."
        },
        {
            "name": "Eliquis",
            "description": "Factor Xa inhibitor anticoagulant",
            "active_ingredients": "Apixaban 2.5mg, 5mg",
            "ai_risk": "High",
            "ai_details": "Risk of bleeding. Requires careful dosing in renal impairment. Regular monitoring needed."
        },
        {
            "name": "Xeljanz",
            "description": "JAK inhibitor for rheumatoid arthritis",
            "active_ingredients": "Tofacitinib 5mg, 10mg",
            "ai_risk": "High",
            "ai_details": "Black box warning for infections, malignancies, and thrombosis. Enhanced monitoring required."
        },
        {
            "name": "Prevnar 13",
            "description": "Pneumococcal vaccine",
            "active_ingredients": "13-valent pneumococcal conjugate vaccine",
            "ai_risk": "Low",
            "ai_details": "Standard vaccine safety profile. Monitor for allergic reactions."
        }
    ],
    "Johnson & Johnson": [
        {
            "name": "Stelara",
            "description": "IL-12/IL-23 inhibitor for psoriasis and Crohn's disease",
            "active_ingredients": "Ustekinumab 45mg, 90mg",
            "ai_risk": "Medium",
            "ai_details": "Monitor for infections and hypersensitivity. Generally well tolerated."
        },
        {
            "name": "Xarelto",
            "description": "Factor Xa inhibitor anticoagulant",
            "active_ingredients": "Rivaroxaban 10mg, 15mg, 20mg",
            "ai_risk": "High",
            "ai_details": "Bleeding risk. No routine monitoring required but careful patient selection needed."
        },
        {
            "name": "Invega Sustenna",
            "description": "Long-acting injectable antipsychotic",
            "active_ingredients": "Paliperidone palmitate",
            "ai_risk": "Medium",
            "ai_details": "Monitor for extrapyramidal symptoms and metabolic changes."
        },
        {
            "name": "Darzalex",
            "description": "CD38-directed antibody for multiple myeloma",
            "active_ingredients": "Daratumumab 100mg, 400mg",
            "ai_risk": "Medium",
            "ai_details": "Infusion reactions common. Monitor blood counts and infections."
        }
    ],
    "Roche Pharmaceuticals": [
        {
            "name": "Avastin",
            "description": "VEGF inhibitor for various cancers",
            "active_ingredients": "Bevacizumab 100mg, 400mg",
            "ai_risk": "High",
            "ai_details": "Black box warnings for GI perforation, hemorrhage, and wound healing. Close monitoring required."
        },
        {
            "name": "Herceptin",
            "description": "HER2-targeted therapy for breast cancer",
            "active_ingredients": "Trastuzumab 150mg, 420mg",
            "ai_risk": "Medium",
            "ai_details": "Cardiac monitoring essential. Risk of cardiomyopathy and infusion reactions."
        },
        {
            "name": "Rituxan",
            "description": "CD20-directed antibody for lymphoma and autoimmune diseases",
            "active_ingredients": "Rituximab 100mg, 500mg",
            "ai_risk": "Medium",
            "ai_details": "Infusion reactions and infection risk. Monitor for PML."
        }
    ],
    "AstraZeneca": [
        {
            "name": "Farxiga",
            "description": "SGLT2 inhibitor for diabetes and heart failure",
            "active_ingredients": "Dapagliflozin 5mg, 10mg",
            "ai_risk": "Medium",
            "ai_details": "Risk of DKA and genital infections. Monitor renal function."
        },
        {
            "name": "Symbicort",
            "description": "Combination inhaler for asthma and COPD",
            "active_ingredients": "Budesonide/Formoterol 80/4.5mcg, 160/4.5mcg",
            "ai_risk": "Low",
            "ai_details": "Standard inhaled corticosteroid risks. Monitor for thrush and systemic effects."
        },
        {
            "name": "Tagrisso",
            "description": "EGFR inhibitor for lung cancer",
            "active_ingredients": "Osimertinib 40mg, 80mg",
            "ai_risk": "High",
            "ai_details": "Risk of ILD, QT prolongation, and cardiomyopathy. Regular monitoring needed."
        }
    ],
    "Merck & Co.": [
        {
            "name": "Januvia",
            "description": "DPP-4 inhibitor for type 2 diabetes",
            "active_ingredients": "Sitagliptin 25mg, 50mg, 100mg",
            "ai_risk": "Low",
            "ai_details": "Generally well tolerated. Monitor for pancreatitis."
        },
        {
            "name": "Keytruda",
            "description": "PD-1 inhibitor immunotherapy for cancer",
            "active_ingredients": "Pembrolizumab 100mg",
            "ai_risk": "High",
            "ai_details": "Immune-related adverse events. Requires intensive monitoring and management."
        },
        {
            "name": "Gardasil 9",
            "description": "HPV vaccine",
            "active_ingredients": "9-valent HPV vaccine",
            "ai_risk": "Low",
            "ai_details": "Standard vaccine safety profile. Monitor for syncope and allergic reactions."
        }
    ],
    "GSK (GlaxoSmithKline)": [
        {
            "name": "Advair",
            "description": "Combination inhaler for asthma and COPD",
            "active_ingredients": "Fluticasone/Salmeterol 100/50mcg, 250/50mcg, 500/50mcg",
            "ai_risk": "Low",
            "ai_details": "Standard ICS/LABA risks. Monitor for pneumonia in COPD patients."
        },
        {
            "name": "Shingrix",
            "description": "Recombinant zoster vaccine",
            "active_ingredients": "Varicella zoster glycoprotein E",
            "ai_risk": "Low",
            "ai_details": "Common reactogenicity but excellent safety profile."
        },
        {
            "name": "Nucala",
            "description": "IL-5 inhibitor for severe asthma",
            "active_ingredients": "Mepolizumab 100mg",
            "ai_risk": "Low",
            "ai_details": "Generally well tolerated. Monitor for hypersensitivity."
        }
    ],
    "Sanofi": [
        {
            "name": "Lantus",
            "description": "Long-acting insulin",
            "active_ingredients": "Insulin glargine 100 units/mL",
            "ai_risk": "Medium",
            "ai_details": "Hypoglycemia risk. Patient education critical."
        },
        {
            "name": "Dupixent",
            "description": "IL-4/IL-13 inhibitor for atopic dermatitis and asthma",
            "active_ingredients": "Dupilumab 200mg, 300mg",
            "ai_risk": "Low",
            "ai_details": "Generally safe. Monitor for conjunctivitis and eosinophilia."
        },
        {
            "name": "Plavix",
            "description": "P2Y12 inhibitor antiplatelet",
            "active_ingredients": "Clopidogrel 75mg, 300mg",
            "ai_risk": "Medium",
            "ai_details": "Bleeding risk. Genetic testing for CYP2C19 may be beneficial."
        }
    ]
}

# Patient names for realistic data
FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
               "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
               "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
               "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
               "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
              "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", "King"]

# Symptoms by risk level
SYMPTOMS_LOW = [
    "Mild headache",
    "Slight dizziness when standing",
    "Occasional nausea",
    "Mild fatigue",
    "Dry mouth",
    "Minor stomach upset",
    "Slight insomnia",
    "Mild constipation"
]

SYMPTOMS_MEDIUM = [
    "Persistent headache and dizziness",
    "Moderate nausea with decreased appetite",
    "Fatigue affecting daily activities",
    "Frequent heart palpitations",
    "Unexplained bruising",
    "Persistent dry cough",
    "Muscle weakness and joint pain",
    "Moderate skin rash with itching"
]

SYMPTOMS_HIGH = [
    "Severe chest pain and difficulty breathing",
    "Extreme dizziness with fainting episodes",
    "Severe allergic reaction with hives and swelling",
    "Irregular heartbeat with chest tightness",
    "Uncontrolled bleeding and bruising",
    "Severe abdominal pain with vomiting",
    "Confusion and memory problems",
    "Severe muscle weakness affecting mobility",
    "High fever with severe headache"
]

# Alert messages
ALERT_TEMPLATES = {
    "Low": [
        "Routine safety update for {drug}: Minor adverse events reported in post-market surveillance.",
        "Information update for {drug}: New drug interaction identified with common antacids.",
        "Quality notification for {drug}: Lot recall due to packaging defect (no safety impact)."
    ],
    "Medium": [
        "Safety alert for {drug}: Increased incidence of mild allergic reactions reported.",
        "Important update for {drug}: New contraindication identified for patients with severe renal impairment.",
        "Pharmacovigilance notice for {drug}: Enhanced monitoring recommended for elderly patients."
    ],
    "High": [
        "URGENT: Safety concern for {drug}: Multiple reports of severe adverse reactions.",
        "Critical alert for {drug}: Potential risk of serious cardiovascular events. Immediate review required.",
        "Important safety information for {drug}: Black box warning being added for hepatotoxicity risk."
    ],
    "Critical": [
        "CRITICAL SAFETY ALERT for {drug}: Voluntary recall initiated due to serious adverse events.",
        "IMMEDIATE ACTION REQUIRED for {drug}: Suspension of sales pending safety investigation.",
        "EMERGENCY NOTIFICATION for {drug}: Life-threatening reactions reported. Urgent patient review needed."
    ]
}

def create_pharma_companies(companies_data):
    """Create pharmaceutical company user accounts"""
    print("\n=== Creating Pharmaceutical Companies ===")
    created_companies = []
    
    for company_data in companies_data:
        # Check if company already exists
        existing = User.query.filter_by(email=company_data["email"]).first()
        if existing:
            print(f"✓ Company already exists: {company_data['name']}")
            created_companies.append(existing)
            continue
        
        company = User(
            name=company_data["name"],
            email=company_data["email"],
            password=company_data["password"],
            role="pharma"
        )
        db.session.add(company)
        created_companies.append(company)
        print(f"✓ Created: {company_data['name']} (Email: {company_data['email']}, Password: {company_data['password']})")
    
    db.session.commit()
    return created_companies

def create_drugs(companies):
    """Create drug portfolios for each company"""
    print("\n=== Creating Drug Portfolios ===")
    created_drugs = []
    
    for company in companies:
        company_name = company.name
        if company_name in DRUG_DATA:
            drugs = DRUG_DATA[company_name]
            print(f"\n{company_name}:")
            
            for drug_data in drugs:
                # Check if drug already exists
                existing = Drug.query.filter_by(
                    name=drug_data["name"],
                    company_id=company.id
                ).first()
                
                if existing:
                    print(f"  ✓ Drug already exists: {drug_data['name']}")
                    created_drugs.append(existing)
                    continue
                
                drug = Drug(
                    name=drug_data["name"],
                    company_id=company.id,
                    description=drug_data["description"],
                    active_ingredients=drug_data["active_ingredients"],
                    ai_risk_assessment=drug_data["ai_risk"],
                    ai_risk_details=drug_data["ai_details"]
                )
                db.session.add(drug)
                created_drugs.append(drug)
                print(f"  ✓ {drug_data['name']} (Risk: {drug_data['ai_risk']})")
    
    db.session.commit()
    return created_drugs

def create_patients(drugs, num_patients=150):
    """Create patient reports (adverse event reports)"""
    print(f"\n=== Creating {num_patients} Patient Reports ===")
    created_patients = []
    
    # Get or create a default doctor
    doctor = User.query.filter_by(role='doctor').first()
    if not doctor:
        doctor = User(
            name="Dr. System Admin",
            email="doctor@system.com",
            password="doctor2024",
            role="doctor"
        )
        db.session.add(doctor)
        db.session.commit()
        print(f"✓ Created default doctor: {doctor.name}")
    
    risk_distribution = {
        'Low': int(num_patients * 0.60),      # 60% low risk
        'Medium': int(num_patients * 0.30),    # 30% medium risk
        'High': int(num_patients * 0.10)       # 10% high risk
    }
    
    patient_count = 0
    for risk_level, count in risk_distribution.items():
        for i in range(count):
            patient_count += 1
            
            # Generate patient data
            first_name = random.choice(FIRST_NAMES)
            last_name = random.choice(LAST_NAMES)
            name = f"{first_name} {last_name}"
            
            # Select appropriate symptoms
            if risk_level == 'Low':
                symptoms = random.choice(SYMPTOMS_LOW)
            elif risk_level == 'Medium':
                symptoms = random.choice(SYMPTOMS_MEDIUM)
            else:
                symptoms = random.choice(SYMPTOMS_HIGH)
            
            # Select random drug
            drug = random.choice(drugs)
            
            # Generate patient ID
            patient_id = f"PT-{random.randint(1000, 9999)}"
            
            # Check if patient ID already exists
            while Patient.query.get(patient_id):
                patient_id = f"PT-{random.randint(1000, 9999)}"
            
            # Random age between 25-85
            age = random.randint(25, 85)
            
            # Gender distribution
            gender = random.choice(['Male', 'Female', 'Male', 'Female', 'Other'])
            
            # Phone number
            phone = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
            
            # Create patient
            patient = Patient(
                id=patient_id,
                created_by=doctor.id,
                name=name,
                phone=phone,
                age=age,
                gender=gender,
                drug_name=drug.name,
                symptoms=symptoms,
                risk_level=risk_level,
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 180))
            )
            
            patient.doctors.append(doctor)
            db.session.add(patient)
            created_patients.append(patient)
            
            if patient_count % 30 == 0:
                print(f"  ✓ Created {patient_count}/{num_patients} patients...")
    
    db.session.commit()
    print(f"✓ Total patients created: {len(created_patients)}")
    return created_patients

def create_alerts(companies, drugs, num_alerts=50):
    """Create safety alerts from pharmaceutical companies"""
    print(f"\n=== Creating {num_alerts} Safety Alerts ===")
    created_alerts = []
    
    severity_distribution = {
        'Low': int(num_alerts * 0.40),      # 40%
        'Medium': int(num_alerts * 0.35),   # 35%
        'High': int(num_alerts * 0.20),     # 20%
        'Critical': int(num_alerts * 0.05)  # 5%
    }
    
    alert_count = 0
    for severity, count in severity_distribution.items():
        for i in range(count):
            alert_count += 1
            
            # Select random company and drug
            company = random.choice(companies)
            drug = random.choice([d for d in drugs if d.company_id == company.id])
            
            # Generate alert message
            template = random.choice(ALERT_TEMPLATES[severity])
            message = template.format(drug=drug.name)
            
            # Create alert
            alert = Alert(
                drug_name=drug.name,
                message=message,
                severity=severity,
                sender_id=company.id,
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 90)),
                is_read=random.choice([True, False, False])  # 1/3 read, 2/3 unread
            )
            
            db.session.add(alert)
            created_alerts.append(alert)
    
    db.session.commit()
    print(f"✓ Total alerts created: {len(created_alerts)}")
    return created_alerts

def export_to_excel(companies, drugs, patients, alerts):
    """Export all data to Excel file"""
    print("\n=== Exporting Data to Excel ===")
    
    # Create Excel writer
    excel_file = 'C:\\Users\\SONUR\\projects\\Novartis\\backend\\pharma_complete_database.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Sheet 1: Companies
        companies_data = [{
            'Company ID': c.id,
            'Company Name': c.name,
            'Email': c.email,
            'Password': c.password,
            'Role': c.role,
            'Number of Drugs': len([d for d in drugs if d.company_id == c.id])
        } for c in companies]
        df_companies = pd.DataFrame(companies_data)
        df_companies.to_excel(writer, sheet_name='Pharma Companies', index=False)
        print("✓ Exported: Pharma Companies sheet")
        
        # Sheet 2: Drugs Portfolio
        drugs_data = [{
            'Drug ID': d.id,
            'Drug Name': d.name,
            'Company Name': d.company.name,
            'Company ID': d.company_id,
            'Description': d.description,
            'Active Ingredients': d.active_ingredients,
            'AI Risk Assessment': d.ai_risk_assessment,
            'AI Risk Details': d.ai_risk_details,
            'Created Date': d.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for d in drugs]
        df_drugs = pd.DataFrame(drugs_data)
        df_drugs.to_excel(writer, sheet_name='Drug Portfolio', index=False)
        print("✓ Exported: Drug Portfolio sheet")
        
        # Sheet 3: Patient Reports
        patients_data = [{
            'Patient ID': p.id,
            'Patient Name': p.name,
            'Phone': p.phone,
            'Age': p.age,
            'Gender': p.gender,
            'Drug Name': p.drug_name,
            'Symptoms': p.symptoms,
            'Risk Level': p.risk_level,
            'Created Date': p.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for p in patients]
        df_patients = pd.DataFrame(patients_data)
        df_patients.to_excel(writer, sheet_name='Patient Reports', index=False)
        print("✓ Exported: Patient Reports sheet")
        
        # Sheet 4: Safety Alerts
        alerts_data = [{
            'Alert ID': a.id,
            'Drug Name': a.drug_name,
            'Message': a.message,
            'Severity': a.severity,
            'Sender Company': a.sender.name,
            'Company ID': a.sender_id,
            'Created Date': a.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'Is Read': 'Yes' if a.is_read else 'No'
        } for a in alerts]
        df_alerts = pd.DataFrame(alerts_data)
        df_alerts.to_excel(writer, sheet_name='Safety Alerts', index=False)
        print("✓ Exported: Safety Alerts sheet")
        
        # Sheet 5: Statistics Summary
        stats_data = [{
            'Metric': 'Total Pharmaceutical Companies',
            'Value': len(companies)
        }, {
            'Metric': 'Total Drugs in Portfolio',
            'Value': len(drugs)
        }, {
            'Metric': 'Total Patient Reports',
            'Value': len(patients)
        }, {
            'Metric': 'High Risk Patients',
            'Value': len([p for p in patients if p.risk_level == 'High'])
        }, {
            'Metric': 'Medium Risk Patients',
            'Value': len([p for p in patients if p.risk_level == 'Medium'])
        }, {
            'Metric': 'Low Risk Patients',
            'Value': len([p for p in patients if p.risk_level == 'Low'])
        }, {
            'Metric': 'Total Safety Alerts',
            'Value': len(alerts)
        }, {
            'Metric': 'Critical Alerts',
            'Value': len([a for a in alerts if a.severity == 'Critical'])
        }, {
            'Metric': 'High Severity Alerts',
            'Value': len([a for a in alerts if a.severity == 'High'])
        }, {
            'Metric': 'Unread Alerts',
            'Value': len([a for a in alerts if not a.is_read])
        }]
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Statistics', index=False)
        print("✓ Exported: Statistics sheet")
    
    print(f"\n✅ Excel file created: {excel_file}")
    return excel_file

def print_summary(companies, drugs, patients, alerts):
    """Print summary of all data"""
    print("\n" + "="*70)
    print("DATABASE POPULATION COMPLETE")
    print("="*70)
    
    print("\n📊 SUMMARY:")
    print(f"  • Pharmaceutical Companies: {len(companies)}")
    print(f"  • Total Drugs: {len(drugs)}")
    print(f"  • Patient Reports: {len(patients)}")
    print(f"    - High Risk: {len([p for p in patients if p.risk_level == 'High'])}")
    print(f"    - Medium Risk: {len([p for p in patients if p.risk_level == 'Medium'])}")
    print(f"    - Low Risk: {len([p for p in patients if p.risk_level == 'Low'])}")
    print(f"  • Safety Alerts: {len(alerts)}")
    print(f"    - Critical: {len([a for a in alerts if a.severity == 'Critical'])}")
    print(f"    - High: {len([a for a in alerts if a.severity == 'High'])}")
    print(f"    - Medium: {len([a for a in alerts if a.severity == 'Medium'])}")
    print(f"    - Low: {len([a for a in alerts if a.severity == 'Low'])}")
    
    print("\n🔐 COMPANY LOGIN CREDENTIALS:")
    print("-" * 70)
    for company_data in PHARMA_COMPANIES:
        print(f"  {company_data['name']}")
        print(f"    Email: {company_data['email']}")
        print(f"    Password: {company_data['password']}")
    
    print("\n" + "="*70)

def main():
    """Main execution function"""
    print("="*70)
    print("PHARMACEUTICAL DATABASE POPULATION SCRIPT")
    print("="*70)
    
    with app.app_context():
        # Create all data
        companies = create_pharma_companies(PHARMA_COMPANIES)
        drugs = create_drugs(companies)
        patients = create_patients(drugs, num_patients=150)
        alerts = create_alerts(companies, drugs, num_alerts=50)
        
        # Export to Excel
        excel_file = export_to_excel(companies, drugs, patients, alerts)
        
        # Print summary
        print_summary(companies, drugs, patients, alerts)
        
        print(f"\n✅ All data has been populated successfully!")
        print(f"📁 Excel file location: {excel_file}")
        print(f"\n🌐 You can now login to any pharma company dashboard using the credentials above.")

if __name__ == '__main__':
    main()
