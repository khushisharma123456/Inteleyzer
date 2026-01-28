"""
Fix passwords for all users to match original setup
"""
import os
os.environ['SKIP_AUTO_POPULATE'] = '1'

from flask import Flask
from models import db, User

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'inteleyzer-secret-key-dev'
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "inteleyzer.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Password mappings
PHARMA_PASSWORDS = {
    "Novartis Pharmaceuticals": "novartis2024",
    "Pfizer Inc.": "pfizer2024",
    "Johnson & Johnson": "jnj2024",
    "Roche Pharmaceuticals": "roche2024",
    "AstraZeneca": "astra2024",
    "Merck & Co.": "merck2024",
    "GSK (GlaxoSmithKline)": "gsk2024",
    "Sanofi": "sanofi2024",
    "Bristol Myers Squibb": "bms2024",
    "Eli Lilly": "lilly2024"
}

HOSPITAL_PASSWORDS = {
    "Central Medical Center": "hospital123",
    "St. Mary's Hospital": "hospital123",
    "City General Hospital": "hospital123"
}

PHARMACY_PASSWORDS = {
    "CVS Pharmacy - Downtown": "pharmacy123",
    "Walgreens - Westside": "pharmacy123",
    "Rite Aid - Eastgate": "pharmacy123",
    "Community Pharmacy": "pharmacy123",
    "HealthMart Pharmacy": "pharmacy123",
    "MedPlus Pharmacy": "pharmacy123",
    "Express Scripts Pharmacy": "pharmacy123",
    "Walmart Pharmacy": "pharmacy123",
    "Costco Pharmacy": "pharmacy123",
    "Target Pharmacy": "pharmacy123"
}

with app.app_context():
    print("Fixing passwords...")
    
    # Fix all doctors
    doctors = User.query.filter_by(role='doctor').all()
    for doc in doctors:
        doc.password = 'doctor123'
    print(f"✓ Fixed {len(doctors)} doctors with password: doctor123")
    
    # Fix pharma companies
    for name, pwd in PHARMA_PASSWORDS.items():
        user = User.query.filter_by(name=name, role='pharma').first()
        if user:
            user.password = pwd
    print(f"✓ Fixed {len(PHARMA_PASSWORDS)} pharma companies")
    
    # Fix hospitals
    for name, pwd in HOSPITAL_PASSWORDS.items():
        user = User.query.filter_by(name=name, role='hospital').first()
        if user:
            user.password = pwd
    print(f"✓ Fixed {len(HOSPITAL_PASSWORDS)} hospitals with password: hospital123")
    
    # Fix pharmacies
    for name, pwd in PHARMACY_PASSWORDS.items():
        user = User.query.filter_by(name=name, role='pharmacy').first()
        if user:
            user.password = pwd
    print(f"✓ Fixed {len(PHARMACY_PASSWORDS)} pharmacies with password: pharmacy123")
    
    db.session.commit()
    print("\n✅ All passwords fixed!")
    print("\nLogin credentials:")
    print("- Doctors: any doctor email + password: doctor123")
    print("- Pharma: company email + password: [company]2024")
    print("- Hospitals: hospital email + password: hospital123")
    print("- Pharmacies: pharmacy email + password: pharmacy123")
