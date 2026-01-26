!/usr/bin/env python3
"""
Populate database with sample patient data for testing
"""
from app import app, db
from models import User, Patient, Drug
import random
from datetime import datetime, timedelta

# Common drugs list
COMMON_DRUGS = [
    'Aspirin',
    'Ibuprofen',
    'Paracetamol',
    'Amoxicillin',
    'Metformin',
    'Lisinopril',
    'Atorvastatin',
    'Omeprazole',
    'Sertraline',
    'Levothyroxine',
    'Amlodipine',
    'Metoprolol',
    'Ciprofloxacin',
    'Azithromycin',
    'Cephalexin',
    'Fluconazole',
    'Loratadine',
    'Cetirizine',
    'Ranitidine',
    'Diclofenac',
    'Naproxen',
    'Tramadol',
    'Codeine',
    'Morphine',
    'Insulin',
    'Glucagon',
    'Warfarin',
    'Heparin',
    'Clopidogrel',
    'Simvastatin',
]

SYMPTOMS = [
    'Mild headache',
    'Nausea',
    'Dizziness',
    'Fatigue',
    'Rash',
    'Itching',
    'Stomach pain',
    'Fever',
    'Cough',
    'Shortness of breath',
    'Chest pain',
    'Severe allergic reaction',
    'Vomiting',
    'Diarrhea',
    'Constipation',
]

GENDERS = ['Male', 'Female', 'Other']
RISK_LEVELS = ['Low', 'Medium', 'High']

def populate_database():
    with app.app_context():
        # Clear existing data
        Patient.query.delete()
        Drug.query.delete()
        User.query.delete()
        
        print("Creating sample users...")
        
        # Create a hospital user
        hospital_user = User(
            name='City Hospital',
            email='hospital@example.com',
            password='password123',
            role='hospital',
            hospital_name='City Hospital'
        )
        db.session.add(hospital_user)
        db.session.commit()
        
        print("Creating sample drugs...")
        
        # Create drug records
        for drug_name in COMMON_DRUGS:
            drug = Drug(
                name=drug_name,
                company_id=1,
                description=f'{drug_name} - Common medication',
                active_ingredients=f'{drug_name} active ingredient',
                ai_risk_assessment=random.choice(['Low', 'Medium', 'High']),
                ai_risk_details='Sample risk assessment'
            )
            db.session.add(drug)
        
        db.session.commit()
        
        print("Creating sample patient records...")
        
        # Create patient records with various drugs
        patient_count = 0
        for drug_name in COMMON_DRUGS:
            # Create 5-15 patients per drug
            num_patients = random.randint(5, 15)
            
            for i in range(num_patients):
                patient = Patient(
                    id=f'PT-{patient_count + 1000}',
                    created_by=hospital_user.id,
                    name=f'Patient {patient_count + 1}',
                    phone=f'555-{random.randint(1000, 9999)}',
                    age=random.randint(18, 85),
                    gender=random.choice(GENDERS),
                    drug_name=drug_name,
                    symptoms=random.choice(SYMPTOMS),
                    risk_level=random.choice(RISK_LEVELS),
                    created_at=datetime.utcnow() - timedelta(days=random.randint(0, 90))
                )
                patient.doctors.append(hospital_user)
                db.session.add(patient)
                patient_count += 1
        
        db.session.commit()
        
        print(f"\n✅ Database populated successfully!")
        print(f"   - Created {patient_count} patient records")
        print(f"   - Created {len(COMMON_DRUGS)} drug records")
        print(f"   - Created 1 hospital user")
        
        # Show sample stats
        print("\nSample statistics:")
        for drug_name in COMMON_DRUGS[:5]:
            patients = Patient.query.filter_by(drug_name=drug_name).all()
            print(f"   {drug_name}: {len(patients)} patients")

if __name__ == '__main__':
    populate_database()
