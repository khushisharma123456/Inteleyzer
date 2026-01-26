"""
Script to populate the database with test data
Run: python populate_test_data.py
"""
import sys
import os

# Set UTF-8 encoding for terminal output
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from pv_backend.app import create_app
from pv_backend.models import db, User, UserRole, Reporter
from datetime import datetime, timedelta

app = create_app()

with app.app_context():
    print("\n=== POPULATING DATABASE WITH TEST DATA ===\n")
    
    # Force create new users
    print("Creating test users...")
    
    # Create Doctor Users
    doctor1 = User(
        email='doctor1@hospital.com',
        full_name='Dr. John Smith',
        role=UserRole.DOCTOR,
        organization='City Hospital',
        license_number='MD-001',
        is_verified=True,
        is_active=True
    )
    doctor1.set_password('password123')
    
    doctor2 = User(
        email='doctor2@hospital.com',
        full_name='Dr. Sarah Johnson',
        role=UserRole.DOCTOR,
        organization='City Hospital',
        license_number='MD-002',
        is_verified=True,
        is_active=True
    )
    doctor2.set_password('password123')
    
    # Create Pharma Users
    hospital1 = User(
        email='hospital1@hospital.com',
        full_name='Hospital Manager Alice',
        role=UserRole.HOSPITAL,
        organization='City Hospital',
        license_number='HOSP-001',
        is_verified=True,
        is_active=True
    )
    hospital1.set_password('password123')
    
    # Create Pharmacy Users
    pharmacy1 = User(
        email='pharmacy1@local.com',
        full_name='Pharmacy Owner Bob',
        role=UserRole.PHARMACY,
        organization='Local Pharmacy',
        license_number='PHARM-001',
        is_verified=True,
        is_active=True
    )
    pharmacy1.set_password('password123')
    
    pharmacy2 = User(
        email='pharmacy2@local.com',
        full_name='Pharmacy Staff Carol',
        role=UserRole.PHARMACY,
        organization='Local Pharmacy',
        license_number='PHARM-002',
        is_verified=False,  # Unverified pharmacy
        is_active=True
    )
    pharmacy2.set_password('password123')
    
    db.session.add_all([doctor1, doctor2, hospital1, pharmacy1, pharmacy2])
    db.session.flush()
    db.session.commit()
    
    print(f"[OK] Created 5 test users")
    print(f"     - 2 Doctors: doctor1@hospital.com, doctor2@hospital.com")
    print(f"     - 1 Hospital: hospital1@hospital.com")
    print(f"     - 2 Pharmacies: pharmacy1@local.com (verified), pharmacy2@local.com (unverified)")
    
    # Create test reporters
    print("\nCreating test reporters...")
    
    reporter1 = Reporter(
        reporter_type='patient',
        full_name='Patient John Doe',
        email='patient1@email.com',
        qualification='',
        institution='Home',
        consent_to_contact=True,
        consent_date=datetime.utcnow()
    )
    reporter2 = Reporter(
        reporter_type='doctor',
        full_name='Dr. Jane Smith',
        email='patient2@email.com',
        qualification='MD',
        institution='City Hospital',
        consent_to_contact=True,
        consent_date=datetime.utcnow()
    )
    db.session.add_all([reporter1, reporter2])
    db.session.flush()
    db.session.commit()
    print(f"[OK] Created 2 test reporters")
    
    print("\n=== TEST DATA POPULATION COMPLETE ===\n")
    
    # Display summary
    print("=== DATABASE SUMMARY ===")
    print(f"Users: {User.query.count()}")
    print(f"Reporters: {Reporter.query.count()}")
    print("\n[LOGIN CREDENTIALS]")
    print("   Doctor:   doctor1@hospital.com / password123")
    print("   Hospital: hospital1@hospital.com / password123")
    print("   Pharmacy: pharmacy1@local.com / password123")

print("\nDatabase seeding completed!")
