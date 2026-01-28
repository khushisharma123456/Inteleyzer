"""
Import data from Excel file back into database
"""
import os
import sys

# Set environment variable to prevent auto-population
os.environ['SKIP_AUTO_POPULATE'] = '1'

from flask import Flask
from models import db, User, Drug, Patient, Alert, hospital_doctor, hospital_drug, hospital_pharmacy, doctor_patient
import pandas as pd
from datetime import datetime

# Create Flask app without auto-population
app = Flask(__name__)
app.config['SECRET_KEY'] = 'inteleyzer-secret-key-dev'
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
os.makedirs(instance_path, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "inteleyzer.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def import_from_excel(excel_file):
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
        
        # Import Users (All roles from Users sheet)
        print("\n=== Importing Users ===")
        if 'Users' in excel_data:
            users_df = excel_data['Users']
            # Create mapping of user names to IDs for later use
            user_name_to_id = {}
            
            for _, row in users_df.iterrows():
                # Set default password as 'password123' since it's not in the Excel
                password = 'password123'
                # Try to extract password from email pattern
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
                # Find company ID from company name
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
        
        # Hospital-Doctor
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
                    pass  # Skip duplicates
            db.session.commit()
            print(f"✓ Imported {len(hd_df)} hospital-doctor relationships")
        
        # Hospital-Drug
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
                    pass  # Skip duplicates
            db.session.commit()
            print(f"✓ Imported {len(hdr_df)} hospital-drug relationships")
        
        # Hospital-Pharmacy
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
                    pass  # Skip duplicates
            db.session.commit()
            print(f"✓ Imported {len(hp_df)} hospital-pharmacy relationships")
        
        db.session.commit()
        
        # Import Patients
        if 'Patients' in excel_data:
            print("\n=== Importing Patients ===")
            patients_df = excel_data['Patients']
            for _, row in patients_df.iterrows():
                # Find created_by user ID from name
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
            
            # Import Doctor-Patient relationships
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
                # Find sender ID from name
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
        
        print("\n" + "="*80)
        print("✅ IMPORT COMPLETE!")
        print("="*80)

if __name__ == '__main__':
    import sys
    excel_file = 'InteLeYzer_Database_20260128_031907.xlsx'
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    import_from_excel(excel_file)
