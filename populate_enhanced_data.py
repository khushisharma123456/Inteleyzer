"""
ENHANCED Database Population Script for Novartis MedSafe Platform
- 500+ Drugs across all companies
- 15-20 Doctors 
- 100+ Patients distributed across doctors
- Hospital accounts with comprehensive data
- Full Excel export
"""

from models import db, User, Drug, Patient, Alert, hospital_doctor, hospital_drug, hospital_pharmacy
from datetime import datetime, timedelta
import random
import pandas as pd

# ============================================================================
# PHARMACEUTICAL COMPANIES
# ============================================================================
PHARMA_COMPANIES = [
    {"name": "Novartis Pharmaceuticals", "email": "admin@novartis.com", "password": "novartis2024"},
    {"name": "Pfizer Inc.", "email": "admin@pfizer.com", "password": "pfizer2024"},
    {"name": "Johnson & Johnson", "email": "admin@jnj.com", "password": "jnj2024"},
    {"name": "Roche Pharmaceuticals", "email": "admin@roche.com", "password": "roche2024"},
    {"name": "AstraZeneca", "email": "admin@astrazeneca.com", "password": "astra2024"},
    {"name": "Merck & Co.", "email": "admin@merck.com", "password": "merck2024"},
    {"name": "GSK (GlaxoSmithKline)", "email": "admin@gsk.com", "password": "gsk2024"},
    {"name": "Sanofi", "email": "admin@sanofi.com", "password": "sanofi2024"},
    {"name": "Bristol Myers Squibb", "email": "admin@bms.com", "password": "bms2024"},
    {"name": "Eli Lilly", "email": "admin@lilly.com", "password": "lilly2024"}
]

# ============================================================================
# DOCTORS (15-20)
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
    {"name": "Dr. William Brown", "email": "w.brown@hospital.com", "password": "doctor123", "specialty": "Pulmonology"},
    {"name": "Dr. Amanda White", "email": "a.white@hospital.com", "password": "doctor123", "specialty": "Pediatrics"},
    {"name": "Dr. Christopher Lee", "email": "c.lee@clinic.com", "password": "doctor123", "specialty": "Orthopedics"},
    {"name": "Dr. Michelle Davis", "email": "m.davis@medcenter.com", "password": "doctor123", "specialty": "Nephrology"},
    {"name": "Dr. Daniel Thompson", "email": "d.thompson@hospital.com", "password": "doctor123", "specialty": "Hematology"},
    {"name": "Dr. Rachel Moore", "email": "r.moore@clinic.com", "password": "doctor123", "specialty": "Infectious Disease"},
    {"name": "Dr. Kevin Harris", "email": "k.harris@medcenter.com", "password": "doctor123", "specialty": "Emergency Medicine"},
    {"name": "Dr. Sophia Clark", "email": "s.clark@hospital.com", "password": "doctor123", "specialty": "Radiology"},
    {"name": "Dr. Nathan Lewis", "email": "n.lewis@clinic.com", "password": "doctor123", "specialty": "Anesthesiology"}
]

# ============================================================================
# HOSPITALS
# ============================================================================
HOSPITALS = [
    {"name": "Central Medical Center", "email": "admin@centralmedcenter.com", "password": "hospital123"},
    {"name": "St. Mary's Hospital", "email": "admin@stmarys.com", "password": "hospital123"},
    {"name": "City General Hospital", "email": "admin@citygeneral.com", "password": "hospital123"}
]

# ============================================================================
# LOCAL PHARMACIES
# ============================================================================
PHARMACIES = [
    {"name": "CVS Pharmacy - Downtown", "email": "downtown@cvs-pharmacy.com", "password": "pharmacy123"},
    {"name": "Walgreens - Westside", "email": "westside@walgreens.com", "password": "pharmacy123"},
    {"name": "Rite Aid - Eastgate", "email": "eastgate@riteaid.com", "password": "pharmacy123"},
    {"name": "Community Pharmacy", "email": "info@communitypharmacy.com", "password": "pharmacy123"},
    {"name": "HealthMart Pharmacy", "email": "contact@healthmart.com", "password": "pharmacy123"},
    {"name": "MedPlus Pharmacy", "email": "info@medplus.com", "password": "pharmacy123"},
    {"name": "Express Scripts Pharmacy", "email": "support@expressscripts.com", "password": "pharmacy123"},
    {"name": "Walmart Pharmacy", "email": "pharmacy@walmart.com", "password": "pharmacy123"},
    {"name": "Costco Pharmacy", "email": "pharmacy@costco.com", "password": "pharmacy123"},
    {"name": "Target Pharmacy", "email": "pharmacy@target.com", "password": "pharmacy123"}
]

# ============================================================================
# DRUG CATEGORIES AND TEMPLATES (500+ drugs)
# ============================================================================
DRUG_CATEGORIES = {
    "Cardiovascular": ["Hypertension", "Heart Failure", "Anticoagulation", "Cholesterol"],
    "Neurology": ["Epilepsy", "Multiple Sclerosis", "Parkinson's", "Alzheimer's"],
    "Oncology": ["Breast Cancer", "Lung Cancer", "Leukemia", "Lymphoma"],
    "Immunology": ["Rheumatoid Arthritis", "Psoriasis", "Crohn's Disease", "Lupus"],
    "Endocrinology": ["Diabetes", "Thyroid", "Osteoporosis", "Growth Hormone"],
    "Psychiatry": ["Depression", "Anxiety", "Schizophrenia", "Bipolar"],
    "Respiratory": ["Asthma", "COPD", "Cystic Fibrosis", "Pulmonary Hypertension"],
    "Gastroenterology": ["GERD", "IBD", "Hepatitis", "IBS"],
    "Infectious Disease": ["HIV", "Hepatitis C", "Antibiotics", "Antifungals"],
    "Vaccines": ["Influenza", "Pneumococcal", "COVID-19", "HPV"]
}

# Patient data
FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
               "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
               "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
               "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
               "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
               "Kevin", "Dorothy", "Brian", "Carol", "George", "Amanda", "Edward", "Melissa",
               "Ronald", "Deborah", "Timothy", "Stephanie", "Jason", "Rebecca", "Jeffrey", "Sharon",
               "Ryan", "Helen", "Jacob", "Laura", "Gary", "Cynthia", "Nicholas", "Amy"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
              "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
              "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams"]

SYMPTOMS_LOW = [
    "Mild headache", "Slight dizziness", "Occasional nausea", "Mild fatigue",
    "Dry mouth", "Minor stomach upset", "Slight insomnia", "Mild constipation"
]

SYMPTOMS_MEDIUM = [
    "Persistent headache and dizziness", "Moderate nausea with decreased appetite",
    "Fatigue affecting daily activities", "Frequent heart palpitations",
    "Unexplained bruising", "Persistent dry cough", "Muscle weakness"
]

SYMPTOMS_HIGH = [
    "Severe chest pain and difficulty breathing", "Extreme dizziness with fainting",
    "Severe allergic reaction with hives", "Irregular heartbeat with chest tightness",
    "Uncontrolled bleeding", "Severe abdominal pain with vomiting"
]

# ============================================================================
# FUNCTIONS
# ============================================================================

def clear_database():
    print("\n=== Clearing Database ===")
    db.session.query(Alert).delete()
    db.session.query(Patient).delete()
    db.session.query(Drug).delete()
    db.session.query(User).delete()
    db.session.commit()
    print("✓ Database cleared")

def create_users():
    print("\n=== Creating User Accounts ===")
    companies = []
    for data in PHARMA_COMPANIES:
        user = User(name=data['name'], email=data['email'], password=data['password'], role='pharma')
        db.session.add(user)
        companies.append(user)
        print(f"✓ Pharma: {data['name']}")
    
    doctors = []
    for data in DOCTORS:
        user = User(name=data['name'], email=data['email'], password=data['password'], role='doctor')
        db.session.add(user)
        doctors.append(user)
        print(f"✓ Doctor: {data['name']} ({data['specialty']})")
    
    hospitals = []
    for data in HOSPITALS:
        user = User(name=data['name'], email=data['email'], password=data['password'], role='hospital')
        db.session.add(user)
        hospitals.append(user)
        print(f"✓ Hospital: {data['name']}")
    
    pharmacies = []
    for data in PHARMACIES:
        user = User(name=data['name'], email=data['email'], password=data['password'], role='pharmacy')
        db.session.add(user)
        pharmacies.append(user)
        print(f"✓ Pharmacy: {data['name']}")
    
    db.session.commit()
    return companies, doctors, hospitals, pharmacies

def create_drugs(companies):
    print(f"\n=== Creating 500+ Drugs ===")
    drugs = []
    drug_counter = 0
    risks = ["Low", "Low", "Low", "Medium", "Medium", "High"]
    
    for company in companies:
        company_drugs = []
        # Each company gets 50-60 drugs
        num_drugs = random.randint(50, 60)
        
        for i in range(num_drugs):
            category = random.choice(list(DRUG_CATEGORIES.keys()))
            indication = random.choice(DRUG_CATEGORIES[category])
            
            drug_name = f"{company.name.split()[0][:4]}-{indication.replace(' ', '')}_{i+1}"
            risk = random.choice(risks)
            
            drug = Drug(
                name=drug_name,
                company_id=company.id,
                description=f"Treatment for {indication}",
                active_ingredients=f"Active compound {drug_counter}",
                ai_risk_assessment=risk,
                ai_risk_details=f"AI assessed {risk} risk profile"
            )
            db.session.add(drug)
            drugs.append(drug)
            company_drugs.append(drug_name)
            drug_counter += 1
        
        print(f"  ✓ {company.name}: {len(company_drugs)} drugs")
        if drug_counter % 100 == 0:
            print(f"  Progress: {drug_counter} drugs created...")
    
    db.session.commit()
    print(f"✓ Total drugs created: {len(drugs)}")
    return drugs

def create_patients(doctors, hospitals, pharmacies, drugs):
    print(f"\n=== Creating 100 Patients ===")
    patients = []
    
    for i in range(100):
        # Randomly assign to doctor
        doctor = random.choice(doctors)
        
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        risk = random.choice(["Low"]*60 + ["Medium"]*30 + ["High"]*10)
        
        if risk == "Low":
            symptoms = random.choice(SYMPTOMS_LOW)
        elif risk == "Medium":
            symptoms = random.choice(SYMPTOMS_MEDIUM)
        else:
            symptoms = random.choice(SYMPTOMS_HIGH)
        
        drug = random.choice(drugs)
        patient_id = f"PT-{random.randint(10000, 99999)}"
        
        while Patient.query.get(patient_id):
            patient_id = f"PT-{random.randint(10000, 99999)}"
        
        patient = Patient(
            id=patient_id,
            created_by=doctor.id,
            name=name,
            phone=f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            age=random.randint(25, 85),
            gender=random.choice(['Male', 'Female', 'Male', 'Female', 'Other']),
            drug_name=drug.name,
            symptoms=symptoms,
            risk_level=risk,
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 180))
        )
        
        patient.doctors.append(doctor)
        # Some patients linked to hospitals too
        if random.random() > 0.5 and hospitals:
            patient.doctors.append(random.choice(hospitals))
        
        db.session.add(patient)
        patients.append(patient)
        
        if (i+1) % 25 == 0:
            print(f"  ✓ Created {i+1}/100 patients...")
    
    db.session.commit()
    print(f"✓ Total patients created: {len(patients)}")
    return patients

def create_alerts(companies, drugs):
    print(f"\n=== Creating Safety Alerts ===")
    alerts = []
    severities = ["Low"]*40 + ["Medium"]*35 + ["High"]*20 + ["Critical"]*5
    
    for i in range(100):
        company = random.choice(companies)
        company_drugs = [d for d in drugs if d.company_id == company.id]
        if not company_drugs:
            continue
        
        drug = random.choice(company_drugs)
        severity = random.choice(severities)
        
        messages = {
            "Low": f"Routine update for {drug.name}: Minor adverse events reported",
            "Medium": f"Safety alert for {drug.name}: Enhanced monitoring recommended",
            "High": f"URGENT: {drug.name} - Multiple serious adverse reactions reported",
            "Critical": f"CRITICAL ALERT: {drug.name} - Immediate action required"
        }
        
        alert = Alert(
            drug_name=drug.name,
            message=messages[severity],
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

def export_to_excel(companies, doctors, hospitals, pharmacies, drugs, patients, alerts):
    print("\n=== Exporting to Excel ===")
    
    excel_file = 'C:\\Users\\SONUR\\projects\\Novartis\\complete_database_enhanced.xlsx'
    
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Companies
            pd.DataFrame([{
                'ID': c.id, 'Name': c.name, 'Email': c.email, 'Password': c.password,
                'Drugs': len([d for d in drugs if d.company_id == c.id])
            } for c in companies]).to_excel(writer, sheet_name='Pharma Companies', index=False)
            
            # Doctors
            pd.DataFrame([{
                'ID': d.id, 'Name': d.name, 'Email': d.email, 'Password': d.password,
                'Patients': len([p for p in patients if d in p.doctors])
            } for d in doctors]).to_excel(writer, sheet_name='Doctors', index=False)
            
            # Hospitals with relationships
            hospital_data = []
            for h in hospitals:
                # Get doctors registered under this hospital
                hospital_doctors = db.session.query(User).join(
                    hospital_doctor, hospital_doctor.c.doctor_id == User.id
                ).filter(
                    hospital_doctor.c.hospital_id == h.id
                ).all()
                
                # Get drugs in use
                hospital_drugs = db.session.query(Drug).join(
                    hospital_drug, hospital_drug.c.drug_id == Drug.id
                ).filter(
                    hospital_drug.c.hospital_id == h.id
                ).all()
                
                # Get pharmacies in contact
                hospital_pharmacies = db.session.query(User).join(
                    hospital_pharmacy, hospital_pharmacy.c.pharmacy_id == User.id
                ).filter(
                    hospital_pharmacy.c.hospital_id == h.id,
                    User.role == 'pharmacy'
                ).all()
                
                hospital_data.append({
                    'ID': h.id,
                    'Name': h.name,
                    'Email': h.email,
                    'Password': h.password,
                    'Doctors': len(hospital_doctors),
                    'Drugs_In_Use': len(hospital_drugs),
                    'Pharmacies': len(hospital_pharmacies)
                })
            
            pd.DataFrame(hospital_data).to_excel(writer, sheet_name='Hospitals', index=False)
            
            # Hospital-Doctor Relationships
            hospital_doctor_data = []
            for h in hospitals:
                hospital_doctors = db.session.query(User).join(
                    hospital_doctor, hospital_doctor.c.doctor_id == User.id
                ).filter(
                    hospital_doctor.c.hospital_id == h.id
                ).all()
                for doc in hospital_doctors:
                    hospital_doctor_data.append({
                        'Hospital_ID': h.id,
                        'Hospital_Name': h.name,
                        'Doctor_ID': doc.id,
                        'Doctor_Name': doc.name,
                        'Doctor_Email': doc.email
                    })
            
            pd.DataFrame(hospital_doctor_data).to_excel(writer, sheet_name='Hospital-Doctors', index=False)
            
            # Hospital-Drug Relationships
            hospital_drug_data = []
            for h in hospitals:
                hospital_drugs = db.session.query(Drug).join(
                    hospital_drug, hospital_drug.c.drug_id == Drug.id
                ).filter(
                    hospital_drug.c.hospital_id == h.id
                ).all()
                for drug in hospital_drugs:
                    hospital_drug_data.append({
                        'Hospital_ID': h.id,
                        'Hospital_Name': h.name,
                        'Drug_ID': drug.id,
                        'Drug_Name': drug.name,
                        'Company': drug.company.name
                    })
            
            pd.DataFrame(hospital_drug_data).to_excel(writer, sheet_name='Hospital-Drugs', index=False)
            
            # Hospital-Pharmacy Relationships
            hospital_pharmacy_data = []
            for h in hospitals:
                hospital_pharmacies = db.session.query(User).join(
                    hospital_pharmacy, hospital_pharmacy.c.pharmacy_id == User.id
                ).filter(
                    hospital_pharmacy.c.hospital_id == h.id
                ).all()
                for pharm in hospital_pharmacies:
                    hospital_pharmacy_data.append({
                        'Hospital_ID': h.id,
                        'Hospital_Name': h.name,
                        'Pharmacy_ID': pharm.id,
                        'Pharmacy_Name': pharm.name,
                        'Pharmacy_Email': pharm.email
                    })
            
            pd.DataFrame(hospital_pharmacy_data).to_excel(writer, sheet_name='Hospital-Pharmacies', index=False)
            
            # Pharmacies
            pd.DataFrame([{
                'ID': p.id, 'Name': p.name, 'Email': p.email, 'Password': p.password
            } for p in pharmacies]).to_excel(writer, sheet_name='Pharmacies', index=False)
            
            # Drugs
            pd.DataFrame([{
                'ID': d.id, 'Name': d.name, 'Company': d.company.name,
                'Description': d.description, 'Risk': d.ai_risk_assessment
            } for d in drugs]).to_excel(writer, sheet_name='Drugs', index=False)
            
            # Patients
            pd.DataFrame([{
                'ID': p.id, 'Name': p.name, 'Age': p.age, 'Gender': p.gender,
                'Phone': p.phone, 'Drug': p.drug_name, 'Symptoms': p.symptoms,
                'Risk': p.risk_level, 'Doctor': User.query.get(p.created_by).name,
                'Date': p.created_at.strftime('%Y-%m-%d')
            } for p in patients]).to_excel(writer, sheet_name='Patients', index=False)
            
            # Alerts
            pd.DataFrame([{
                'ID': a.id, 'Drug': a.drug_name, 'Message': a.message,
                'Severity': a.severity, 'Company': a.sender.name,
                'Date': a.created_at.strftime('%Y-%m-%d')
            } for a in alerts]).to_excel(writer, sheet_name='Alerts', index=False)
            
            # Summary
            pd.DataFrame([
                {'Metric': 'Pharma Companies', 'Value': len(companies)},
                {'Metric': 'Doctors', 'Value': len(doctors)},
                {'Metric': 'Hospitals', 'Value': len(hospitals)},
                {'Metric': 'Pharmacies', 'Value': len(pharmacies)},
                {'Metric': 'Total Drugs', 'Value': len(drugs)},
                {'Metric': 'Total Patients', 'Value': len(patients)},
                {'Metric': 'High Risk Patients', 'Value': len([p for p in patients if p.risk_level == 'High'])},
                {'Metric': 'Total Alerts', 'Value': len(alerts)},
                {'Metric': 'Critical Alerts', 'Value': len([a for a in alerts if a.severity == 'Critical'])}
            ]).to_excel(writer, sheet_name='Summary', index=False)
            
            # All Login Credentials
            all_users = companies + doctors + hospitals + pharmacies
            pd.DataFrame([{
                'Role': u.role.upper(), 'Name': u.name, 'Email': u.email, 'Password': u.password
            } for u in all_users]).to_excel(writer, sheet_name='All Credentials', index=False)
        
        print(f"✓ Excel exported: {excel_file}")
        return excel_file
    except PermissionError:
        print(f"⚠ Could not create Excel file (file may be open). Data is saved in database.")
        return None
    
    print(f"✓ Excel exported: {excel_file}")
    return excel_file

def print_summary(companies, doctors, hospitals, pharmacies, drugs, patients, alerts):
    print("\n" + "="*80)
    print("DATABASE POPULATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 SUMMARY:")
    print(f"  • Pharma Companies: {len(companies)}")
    print(f"  • Doctors: {len(doctors)}")
    print(f"  • Hospitals: {len(hospitals)}")
    print(f"  • Pharmacies: {len(pharmacies)}")
    print(f"  • Total Drugs: {len(drugs)}")
    print(f"  • Total Patients: {len(patients)}")
    print(f"    - High Risk: {len([p for p in patients if p.risk_level == 'High'])}")
    print(f"    - Medium Risk: {len([p for p in patients if p.risk_level == 'Medium'])}")
    print(f"    - Low Risk: {len([p for p in patients if p.risk_level == 'Low'])}")
    print(f"  • Total Alerts: {len(alerts)}")
    
    print(f"\n🔐 SAMPLE CREDENTIALS:")
    print(f"  Pharma: admin@novartis.com / novartis2024")
    print(f"  Doctor: emily.chen@hospital.com / doctor123")
    print(f"  Hospital: admin@centralmedcenter.com / hospital123")
    print(f"  Pharmacy: downtown@cvs-pharmacy.com / pharmacy123")
    print(f"\n🌐 Login at: http://127.0.0.1:5000/login")

def create_hospital_relationships(hospitals, doctors, drugs, pharmacies):
    """Create relationships between hospitals and doctors/drugs/pharmacies"""
    print("\n=== Creating Hospital Relationships ===")
    
    # Distribute doctors evenly across hospitals
    doctors_per_hospital = len(doctors) // len(hospitals)
    
    for i, hospital in enumerate(hospitals):
        # Calculate start and end index for this hospital
        start_idx = i * doctors_per_hospital
        if i == len(hospitals) - 1:
            # Last hospital gets remaining doctors
            end_idx = len(doctors)
        else:
            end_idx = start_idx + doctors_per_hospital
        
        # Assign doctors to hospital
        assigned_doctors = doctors[start_idx:end_idx]
        for doctor in assigned_doctors:
            db.session.execute(
                hospital_doctor.insert().values(
                    hospital_id=hospital.id,
                    doctor_id=doctor.id
                )
            )
        
        print(f"  ✓ {hospital.name}: {len(assigned_doctors)} doctors")
    
    # Assign drugs to hospitals (each hospital uses 50-100 random drugs)
    for hospital in hospitals:
        num_drugs = random.randint(50, min(100, len(drugs)))
        hospital_drugs = random.sample(drugs, num_drugs)
        for drug in hospital_drugs:
            db.session.execute(
                hospital_drug.insert().values(
                    hospital_id=hospital.id,
                    drug_id=drug.id
                )
            )
    
    # Assign pharmacies to hospitals (each hospital has 3-5 pharmacy contacts)
    for hospital in hospitals:
        num_pharmacies = random.randint(3, min(5, len(pharmacies)))
        hospital_pharmacies = random.sample(pharmacies, num_pharmacies)
        for pharmacy in hospital_pharmacies:
            db.session.execute(
                hospital_pharmacy.insert().values(
                    hospital_id=hospital.id,
                    pharmacy_id=pharmacy.id
                )
            )
    
    db.session.commit()
    
    print(f"✓ Hospital relationships created:")
    print(f"  - Each hospital has 50-100 drugs in use")
    print(f"  - Each hospital has 3-5 pharmacy contacts")

def import_from_excel(app, db, excel_file):
    """Import data from Excel file"""
    print("="*80)
    print(f"IMPORTING DATA FROM: {excel_file}")
    print("="*80)
    
    with app.app_context():
        # Clear existing database
        print("\n=== Clearing Database ===")
        db.session.query(Alert).delete()
        db.session.query(Patient).delete()
        db.session.query(Drug).delete()
        db.session.query(User).delete()
        db.session.commit()
        print("✓ Database cleared")
        
        # Read Excel sheets
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        
        # Import Users
        print("\n=== Importing Users ===")
        if 'Users' in excel_data:
            users_df = excel_data['Users']
            user_name_to_id = {}
            
            for _, row in users_df.iterrows():
                password = 'password123'
                if '@' in row['Email']:
                    password = row['Email'].split('@')[0].replace('.', '').lower() + '2024'
                
                user = User(
                    id=int(row['ID']),
                    name=row['Name'],
                    email=row['Email'],
                    password=password,
                    role=row['Role'],
                    hospital_name=row['Hospital Name'] if pd.notna(row.get('Hospital Name')) else None
                )
                db.session.add(user)
                user_name_to_id[row['Name']] = int(row['ID'])
            
            db.session.commit()
            pharma_count = len(users_df[users_df['Role'] == 'pharma'])
            doctor_count = len(users_df[users_df['Role'] == 'doctor'])
            hospital_count = len(users_df[users_df['Role'] == 'hospital'])
            pharmacy_count = len(users_df[users_df['Role'] == 'pharmacy'])
            
            print(f"✓ Imported {len(users_df)} users:")
            print(f"  - {pharma_count} pharma companies")
            print(f"  - {doctor_count} doctors")
            print(f"  - {hospital_count} hospitals")
            print(f"  - {pharmacy_count} pharmacies")
        
        # Import Drugs
        if 'Drugs' in excel_data:
            print("\n=== Importing Drugs ===")
            drugs_df = excel_data['Drugs']
            for _, row in drugs_df.iterrows():
                company_name = row['Company']
                company_id = user_name_to_id.get(company_name)
                
                if company_id:
                    drug = Drug(
                        id=int(row['ID']),
                        name=row['Name'],
                        company_id=company_id,
                        description=row.get('Description', ''),
                        active_ingredients=row.get('Active Ingredients', ''),
                        ai_risk_assessment=row.get('AI Risk Assessment', 'Analyzing'),
                        ai_risk_details=row.get('AI Risk Details', '')
                    )
                    db.session.add(drug)
            db.session.commit()
            print(f"✓ Imported {len(drugs_df)} drugs")
        
        # Import Hospital Relationships
        print("\n=== Importing Hospital Relationships ===")
        from models import hospital_doctor, hospital_drug, hospital_pharmacy, doctor_patient
        
        if 'Hospital-Doctor Links' in excel_data:
            hd_df = excel_data['Hospital-Doctor Links']
            for _, row in hd_df.iterrows():
                try:
                    db.session.execute(
                        hospital_doctor.insert().values(
                            hospital_id=int(row['Hospital ID']),
                            doctor_id=int(row['Doctor ID'])
                        )
                    )
                except:
                    pass
            db.session.commit()
            print(f"✓ Imported {len(hd_df)} hospital-doctor relationships")
        
        if 'Hospital-Drug Links' in excel_data:
            hdr_df = excel_data['Hospital-Drug Links']
            for _, row in hdr_df.iterrows():
                try:
                    db.session.execute(
                        hospital_drug.insert().values(
                            hospital_id=int(row['Hospital ID']),
                            drug_id=int(row['Drug ID'])
                        )
                    )
                except:
                    pass
            db.session.commit()
            print(f"✓ Imported {len(hdr_df)} hospital-drug relationships")
        
        if 'Hospital-Pharmacy Links' in excel_data:
            hp_df = excel_data['Hospital-Pharmacy Links']
            for _, row in hp_df.iterrows():
                try:
                    db.session.execute(
                        hospital_pharmacy.insert().values(
                            hospital_id=int(row['Hospital ID']),
                            pharmacy_id=int(row['Pharmacy ID'])
                        )
                    )
                except:
                    pass
            db.session.commit()
            print(f"✓ Imported {len(hp_df)} hospital-pharmacy relationships")
        
        # Import Patients
        if 'Patients' in excel_data:
            print("\n=== Importing Patients ===")
            patients_df = excel_data['Patients']
            for _, row in patients_df.iterrows():
                created_by_name = row.get('Created By')
                created_by_id = user_name_to_id.get(created_by_name) if pd.notna(created_by_name) else None
                
                patient = Patient(
                    id=row['ID'],
                    name=row['Name'],
                    phone=str(row['Phone']) if pd.notna(row['Phone']) else None,
                    age=int(row['Age']),
                    gender=row['Gender'],
                    drug_name=row['Drug Name'],
                    symptoms=row.get('Symptoms', ''),
                    risk_level=row.get('Risk Level', 'Low'),
                    case_status=row.get('Case Status', 'Active'),
                    created_by=created_by_id
                )
                db.session.add(patient)
            db.session.commit()
            print(f"✓ Imported {len(patients_df)} patients")
            
            if 'Doctor-Patient Links' in excel_data:
                dp_df = excel_data['Doctor-Patient Links']
                for _, row in dp_df.iterrows():
                    try:
                        db.session.execute(
                            doctor_patient.insert().values(
                                doctor_id=int(row['Doctor ID']),
                                patient_id=row['Patient ID']
                            )
                        )
                    except:
                        pass
                db.session.commit()
                print(f"✓ Imported {len(dp_df)} doctor-patient relationships")
        
        # Import Alerts
        if 'Alerts' in excel_data:
            print("\n=== Importing Alerts ===")
            alerts_df = excel_data['Alerts']
            for _, row in alerts_df.iterrows():
                sender_name = row.get('Sender')
                sender_id = user_name_to_id.get(sender_name)
                drug_name = row.get('Drug Name')
                
                if sender_id and drug_name:
                    try:
                        alert = Alert(
                            id=int(row['ID']),
                            drug_name=drug_name,
                            title=row.get('Title') if pd.notna(row.get('Title')) else None,
                            message=row['Message'],
                            severity=row.get('Severity', 'Medium'),
                            sender_id=sender_id,
                            recipient_type=row.get('Recipient Type', 'all'),
                            is_read=True if row.get('Is Read') == 'Yes' else False,
                            created_at=pd.to_datetime(row['Created At']) if pd.notna(row.get('Created At')) else datetime.utcnow()
                        )
                        db.session.add(alert)
                    except Exception as e:
                        print(f"  ! Skipping alert {row.get('ID')}: {str(e)}")
            db.session.commit()
            print(f"✓ Imported alerts")
        
        # Fix passwords after import
        print("\n=== Fixing Passwords ===")
        users = User.query.all()
        for user in users:
            if user.role == 'doctor':
                user.password = 'doctor123'
            elif user.role == 'pharma':
                company_name = user.email.split('@')[0]
                user.password = f'{company_name}2024'
            elif user.role == 'hospital':
                user.password = 'hospital123'
            elif user.role == 'pharmacy':
                user.password = 'pharmacy123'
        db.session.commit()
        print("✓ Passwords fixed")
        
        print("\n" + "="*80)
        print("✅ IMPORT COMPLETE!")
        print("="*80)


def populate_database(app, db):
    """Main function to populate database - checks for Excel first, otherwise generates new data"""
    import os
    import glob
    
    # Check for existing Excel files (most recent first)
    excel_files = sorted(glob.glob('InteLeYzer_Database_*.xlsx'), reverse=True)
    
    if excel_files:
        print(f"\n📁 Found Excel backup: {excel_files[0]}")
        print("Importing data from Excel file...")
        import_from_excel(app, db, excel_files[0])
    else:
        print("="*80)
        print("ENHANCED DATABASE POPULATION - Generating New Data")
        print("="*80)
        
        with app.app_context():
            clear_database()
            companies, doctors, hospitals, pharmacies = create_users()
            drugs = create_drugs(companies)
            
            # Create hospital relationships before patients
            create_hospital_relationships(hospitals, doctors, drugs, pharmacies)
            
            patients = create_patients(doctors, hospitals, pharmacies, drugs)
            alerts = create_alerts(companies, drugs)
            excel_file = export_to_excel(companies, doctors, hospitals, pharmacies, drugs, patients, alerts)
            print_summary(companies, doctors, hospitals, pharmacies, drugs, patients, alerts)
            print(f"\n✅ Complete! Excel: {excel_file}")

if __name__ == '__main__':
    from app import app, db
    populate_database(app, db)
