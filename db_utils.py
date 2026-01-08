"""
🗄️ DATABASE UTILITIES FOR PATIENT DATA RETRIEVAL

This module provides database connection and query functions for the
Pharmacovigilance WhatsApp Agent to retrieve patient data on request.

PRIVACY NOTES:
- Patients can ONLY access their own data (identified by phone number)
- No cross-patient data access is allowed
- All queries are parameterized to prevent SQL injection

Author: Generated for patient data query feature
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Database type: 'mysql', 'postgresql', or 'sqlite'
DB_TYPE = os.getenv('DB_TYPE', 'mysql')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '3306'))
DB_NAME = os.getenv('DB_NAME', 'pharmacovigilance')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# SQLite database file path (for development/testing)
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'dataset/pharmacovigilance.db')


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_connection():
    """
    Create and return a database connection based on configured DB_TYPE.
    
    Returns:
        Database connection object, or None if connection fails
    """
    try:
        if DB_TYPE == 'mysql':
            import mysql.connector
            conn = mysql.connector.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info("✅ Connected to MySQL database")
            return conn
            
        elif DB_TYPE == 'postgresql':
            import psycopg2
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info("✅ Connected to PostgreSQL database")
            return conn
            
        elif DB_TYPE == 'sqlite':
            import sqlite3
            conn = sqlite3.connect(SQLITE_DB_PATH)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"✅ Connected to SQLite database: {SQLITE_DB_PATH}")
            return conn
            
        else:
            logger.error(f"❌ Unsupported database type: {DB_TYPE}")
            return None
            
    except ImportError as e:
        logger.error(f"❌ Database driver not installed: {e}")
        logger.info("Install with: pip install mysql-connector-python (for MySQL)")
        logger.info("Install with: pip install psycopg2-binary (for PostgreSQL)")
        return None
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return None


def close_db_connection(conn):
    """Safely close a database connection."""
    if conn:
        try:
            conn.close()
            logger.info("✅ Database connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing connection: {e}")


# =============================================================================
# PATIENT DATA QUERIES
# =============================================================================

def get_patient_by_phone(phone_number: str) -> Optional[Dict[str, Any]]:
    """
    Get patient information by phone number.
    
    Args:
        phone_number: Patient's phone number (e.g., '+919876543210')
        
    Returns:
        Dict with patient info, or None if not found
    """
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cursor = conn.cursor()
        query = """
            SELECT patient_id, name, phone_number, age, gender, created_at
            FROM Patient
            WHERE phone_number = %s
        """ if DB_TYPE != 'sqlite' else """
            SELECT patient_id, name, phone_number, age, gender, created_at
            FROM Patient
            WHERE phone_number = ?
        """
        
        cursor.execute(query, (phone_number,))
        row = cursor.fetchone()
        
        if row:
            if DB_TYPE == 'sqlite':
                return dict(row)
            else:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        return None
        
    except Exception as e:
        logger.error(f"❌ Error querying patient: {e}")
        return None
        
    finally:
        close_db_connection(conn)


def get_patient_visits(patient_id: str) -> List[Dict[str, Any]]:
    """
    Get all visits for a patient.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        List of visit dictionaries
    """
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        cursor = conn.cursor()
        query = """
            SELECT v.visit_id, v.visit_date, v.visit_type, v.disease_name, v.status,
                   d.name as doctor_name, d.department
            FROM Visit v
            LEFT JOIN Doctor d ON v.doctor_id = d.doctor_id
            WHERE v.patient_id = %s
            ORDER BY v.visit_date DESC
        """ if DB_TYPE != 'sqlite' else """
            SELECT v.visit_id, v.visit_date, v.visit_type, v.disease_name, v.status,
                   d.name as doctor_name, d.department
            FROM Visit v
            LEFT JOIN Doctor d ON v.doctor_id = d.doctor_id
            WHERE v.patient_id = ?
            ORDER BY v.visit_date DESC
        """
        
        cursor.execute(query, (patient_id,))
        rows = cursor.fetchall()
        
        if DB_TYPE == 'sqlite':
            return [dict(row) for row in rows]
        else:
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
            
    except Exception as e:
        logger.error(f"❌ Error querying visits: {e}")
        return []
        
    finally:
        close_db_connection(conn)


def get_visit_details(visit_id: int) -> Optional[Dict[str, Any]]:
    """
    Get detailed information for a specific visit including doctor entry and responses.
    
    Args:
        visit_id: Visit identifier
        
    Returns:
        Dict with visit details, or None if not found
    """
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cursor = conn.cursor()
        
        # Get visit basic info
        visit_query = """
            SELECT v.*, d.name as doctor_name, d.department
            FROM Visit v
            LEFT JOIN Doctor d ON v.doctor_id = d.doctor_id
            WHERE v.visit_id = %s
        """ if DB_TYPE != 'sqlite' else """
            SELECT v.*, d.name as doctor_name, d.department
            FROM Visit v
            LEFT JOIN Doctor d ON v.doctor_id = d.doctor_id
            WHERE v.visit_id = ?
        """
        
        cursor.execute(visit_query, (visit_id,))
        visit_row = cursor.fetchone()
        
        if not visit_row:
            return None
            
        if DB_TYPE == 'sqlite':
            visit_data = dict(visit_row)
        else:
            columns = [desc[0] for desc in cursor.description]
            visit_data = dict(zip(columns, visit_row))
        
        # Get doctor entry (prescription)
        entry_query = """
            SELECT medicine_prescribed, dosage, frequency, duration, 
                   diagnosis_notes, additional_instructions
            FROM Doctor_Entry
            WHERE visit_id = %s
        """ if DB_TYPE != 'sqlite' else """
            SELECT medicine_prescribed, dosage, frequency, duration, 
                   diagnosis_notes, additional_instructions
            FROM Doctor_Entry
            WHERE visit_id = ?
        """
        
        cursor.execute(entry_query, (visit_id,))
        entry_row = cursor.fetchone()
        
        if entry_row:
            if DB_TYPE == 'sqlite':
                visit_data['prescription'] = dict(entry_row)
            else:
                columns = [desc[0] for desc in cursor.description]
                visit_data['prescription'] = dict(zip(columns, entry_row))
        
        return visit_data
        
    except Exception as e:
        logger.error(f"❌ Error querying visit details: {e}")
        return None
        
    finally:
        close_db_connection(conn)


def get_patient_responses(visit_id: int) -> List[Dict[str, Any]]:
    """
    Get all patient responses for a visit.
    
    Args:
        visit_id: Visit identifier
        
    Returns:
        List of response dictionaries
    """
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        cursor = conn.cursor()
        query = """
            SELECT question_id, answer, language, response_timestamp
            FROM Patient_Response
            WHERE visit_id = %s
            ORDER BY response_timestamp ASC
        """ if DB_TYPE != 'sqlite' else """
            SELECT question_id, answer, language, response_timestamp
            FROM Patient_Response
            WHERE visit_id = ?
            ORDER BY response_timestamp ASC
        """
        
        cursor.execute(query, (visit_id,))
        rows = cursor.fetchall()
        
        if DB_TYPE == 'sqlite':
            return [dict(row) for row in rows]
        else:
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
            
    except Exception as e:
        logger.error(f"❌ Error querying responses: {e}")
        return []
        
    finally:
        close_db_connection(conn)


def get_quality_report(visit_id: int) -> Optional[Dict[str, Any]]:
    """
    Get data quality report for a visit.
    
    Args:
        visit_id: Visit identifier
        
    Returns:
        Dict with quality report, or None if not found
    """
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cursor = conn.cursor()
        query = """
            SELECT data_quality_status, confidence_score, 
                   missing_fields, inconsistent_fields, clarification_required
            FROM Data_Quality_Report
            WHERE visit_id = %s
        """ if DB_TYPE != 'sqlite' else """
            SELECT data_quality_status, confidence_score, 
                   missing_fields, inconsistent_fields, clarification_required
            FROM Data_Quality_Report
            WHERE visit_id = ?
        """
        
        cursor.execute(query, (visit_id,))
        row = cursor.fetchone()
        
        if row:
            if DB_TYPE == 'sqlite':
                return dict(row)
            else:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        return None
        
    except Exception as e:
        logger.error(f"❌ Error querying quality report: {e}")
        return None
        
    finally:
        close_db_connection(conn)


def get_patient_full_data(phone_number: str) -> Optional[Dict[str, Any]]:
    """
    Get all data for a patient identified by phone number.
    This is the main function called when patient asks "show my data".
    
    Args:
        phone_number: Patient's phone number
        
    Returns:
        Complete patient data dictionary, or None if not found
    """
    # Get patient info
    patient = get_patient_by_phone(phone_number)
    if not patient:
        logger.info(f"No patient found for phone: {phone_number}")
        return None
    
    patient_id = patient['patient_id']
    
    # Get all visits
    visits = get_patient_visits(patient_id)
    
    # For each visit, get detailed info
    for visit in visits:
        visit_id = visit['visit_id']
        
        # Get prescription/doctor entry
        details = get_visit_details(visit_id)
        if details and 'prescription' in details:
            visit['prescription'] = details['prescription']
        
        # Get patient responses
        visit['responses'] = get_patient_responses(visit_id)
        
        # Get quality report
        visit['quality_report'] = get_quality_report(visit_id)
    
    return {
        'patient': patient,
        'visits': visits
    }


# =============================================================================
# WHATSAPP MESSAGE FORMATTING
# =============================================================================

# Question ID to human-readable label mapping
QUESTION_LABELS = {
    'Q1_language': 'Language',
    'Q2_medicine_started': 'Medicine Started',
    'Q2_medicine_continued': 'Medicine Continued',
    'Q3_adherence': 'Adherence',
    'Q4_food_relation': 'Food Relation',
    'Q5_previous_symptoms': 'Previous Symptoms',
    'Q5_new_symptoms': 'New Symptoms',
    'Q6_overall_feeling': 'Overall Feeling',
    'Q7_new_symptoms': 'New Symptoms',
    'Q8_symptom_description': 'Symptom Description',
    'Q9_onset': 'Symptom Onset',
    'Q9_safety_check': 'Safety Check',
    'Q10_severity': 'Severity',
    'Q11_body_parts': 'Body Parts Affected',
    'Q12_safety_check': 'Safety Check'
}


def format_patient_data_for_whatsapp(data: Dict[str, Any]) -> str:
    """
    Format patient data into a WhatsApp-friendly message.
    
    Args:
        data: Patient data dictionary from get_patient_full_data()
        
    Returns:
        Formatted string for WhatsApp message
    """
    if not data:
        return "❌ No data found for your phone number."
    
    patient = data.get('patient', {})
    visits = data.get('visits', [])
    
    # Build message
    lines = []
    
    # Patient header
    lines.append("📋 *Your Medical Records*")
    lines.append("")
    
    # Patient info
    lines.append("👤 *Patient Information*")
    lines.append(f"Name: {patient.get('name', 'N/A')}")
    lines.append(f"Patient ID: {patient.get('patient_id', 'N/A')}")
    lines.append(f"Age: {patient.get('age', 'N/A')}")
    lines.append(f"Gender: {patient.get('gender', 'N/A').capitalize() if patient.get('gender') else 'N/A'}")
    lines.append("")
    
    if not visits:
        lines.append("📅 No visits on record.")
    else:
        # Show latest visit (or all if requested)
        for i, visit in enumerate(visits[:3]):  # Limit to 3 most recent
            lines.append("─" * 20)
            lines.append(f"📅 *Visit {i+1}*")
            lines.append(f"Visit ID: {visit.get('visit_id', 'N/A')}")
            lines.append(f"Date: {visit.get('visit_date', 'N/A')}")
            lines.append(f"Type: {visit.get('visit_type', 'N/A').replace('_', ' ').title() if visit.get('visit_type') else 'N/A'}")
            lines.append(f"Disease: {visit.get('disease_name', 'N/A')}")
            lines.append(f"Doctor: {visit.get('doctor_name', 'N/A')}")
            lines.append(f"Department: {visit.get('department', 'N/A')}")
            lines.append(f"Status: {visit.get('status', 'N/A').replace('_', ' ').title() if visit.get('status') else 'N/A'}")
            lines.append("")
            
            # Prescription
            prescription = visit.get('prescription', {})
            if prescription:
                lines.append("💊 *Prescribed Medicine*")
                lines.append(f"Medicine: {prescription.get('medicine_prescribed', 'N/A')}")
                lines.append(f"Dosage: {prescription.get('dosage', 'N/A')}")
                lines.append(f"Frequency: {prescription.get('frequency', 'N/A')}")
                lines.append(f"Duration: {prescription.get('duration', 'N/A')}")
                if prescription.get('additional_instructions'):
                    lines.append(f"Instructions: {prescription['additional_instructions']}")
                lines.append("")
            
            # Responses (summarized)
            responses = visit.get('responses', [])
            if responses:
                lines.append("📝 *Your Responses*")
                for resp in responses:
                    question_id = resp.get('question_id', '')
                    label = QUESTION_LABELS.get(question_id, question_id.replace('_', ' ').title())
                    answer = resp.get('answer', 'N/A')
                    lines.append(f"• {label}: {answer}")
                lines.append("")
            
            # Quality report
            quality = visit.get('quality_report')
            if quality:
                status = quality.get('data_quality_status', 'N/A')
                score = quality.get('confidence_score', 0)
                status_emoji = '✅' if status == 'PASS' else '⚠️'
                lines.append(f"📊 *Data Quality*: {status_emoji} {status} (Score: {score})")
                lines.append("")
    
    # Footer
    lines.append("─" * 20)
    lines.append("💬 Reply with a number to continue your follow-up,")
    lines.append("or type 'help' for assistance.")
    
    return "\n".join(lines)


def format_visit_details_for_whatsapp(visit: Dict[str, Any]) -> str:
    """
    Format a single visit's details for WhatsApp.
    
    Args:
        visit: Visit data dictionary
        
    Returns:
        Formatted string for WhatsApp message
    """
    if not visit:
        return "❌ Visit not found."
    
    lines = []
    lines.append(f"📅 *Visit Details - ID: {visit.get('visit_id', 'N/A')}*")
    lines.append("")
    lines.append(f"Date: {visit.get('visit_date', 'N/A')}")
    lines.append(f"Type: {visit.get('visit_type', 'N/A')}")
    lines.append(f"Disease: {visit.get('disease_name', 'N/A')}")
    lines.append(f"Status: {visit.get('status', 'N/A')}")
    lines.append(f"Doctor: {visit.get('doctor_name', 'N/A')}")
    
    prescription = visit.get('prescription', {})
    if prescription:
        lines.append("")
        lines.append("💊 *Prescription*")
        lines.append(f"Medicine: {prescription.get('medicine_prescribed', 'N/A')}")
        lines.append(f"Dosage: {prescription.get('dosage', 'N/A')}")
        lines.append(f"Frequency: {prescription.get('frequency', 'N/A')}")
        lines.append(f"Duration: {prescription.get('duration', 'N/A')}")
    
    return "\n".join(lines)


# =============================================================================
# INTENT DETECTION FOR DATA QUERIES
# =============================================================================

# Keywords that indicate a data query request
DATA_QUERY_KEYWORDS = [
    # Full data requests
    'show my data', 'show my records', 'show my information',
    'view my data', 'view my records', 'view my information',
    'get my data', 'get my records', 'see my data',
    'my data', 'my records', 'my information',
    'mera data', 'meri information',  # Hindi
    
    # Specific requests
    'my visits', 'my appointments', 'my visit history',
    'my prescriptions', 'my medicines', 'my medication',
    'my treatment', 'my responses', 'my answers',
    
    # Questions
    'what is my data', 'what are my records',
    'can i see my data', 'can you show my data',
]

# Keywords for specific visit queries
VISIT_QUERY_KEYWORDS = [
    'visit id', 'visit number', 'visit #',
    'show visit', 'get visit', 'view visit',
]


def detect_data_query_intent(text: str) -> Dict[str, Any]:
    """
    Detect if the user's message is requesting their data.
    
    Args:
        text: User's input message
        
    Returns:
        Dict with:
        - is_data_query: bool - whether this is a data query
        - query_type: str - 'full_data', 'visit_details', or None
        - visit_id: int - specific visit ID if mentioned, else None
    """
    text_lower = text.lower().strip()
    
    result = {
        'is_data_query': False,
        'query_type': None,
        'visit_id': None
    }
    
    # Check for full data query
    for keyword in DATA_QUERY_KEYWORDS:
        if keyword in text_lower:
            result['is_data_query'] = True
            result['query_type'] = 'full_data'
            return result
    
    # Check for specific visit query
    for keyword in VISIT_QUERY_KEYWORDS:
        if keyword in text_lower:
            result['is_data_query'] = True
            result['query_type'] = 'visit_details'
            
            # Try to extract visit ID number
            import re
            numbers = re.findall(r'\d+', text_lower)
            if numbers:
                result['visit_id'] = int(numbers[0])
            
            return result
    
    return result


# =============================================================================
# MAIN HANDLER FOR DATA QUERIES
# =============================================================================

def handle_data_query(phone_number: str, query_type: str = 'full_data', 
                      visit_id: int = None) -> str:
    """
    Main handler for patient data queries.
    
    Args:
        phone_number: Patient's phone number
        query_type: Type of query ('full_data' or 'visit_details')
        visit_id: Specific visit ID for detailed queries
        
    Returns:
        Formatted WhatsApp message with patient data
    """
    logger.info(f"📊 Data query: phone={phone_number}, type={query_type}, visit_id={visit_id}")
    
    if query_type == 'full_data':
        data = get_patient_full_data(phone_number)
        if data:
            return format_patient_data_for_whatsapp(data)
        else:
            return (
                "❌ *No Records Found*\n\n"
                "We couldn't find any records associated with your phone number.\n\n"
                "If you believe this is an error, please contact your healthcare provider."
            )
    
    elif query_type == 'visit_details' and visit_id:
        # First verify the visit belongs to this patient
        patient = get_patient_by_phone(phone_number)
        if not patient:
            return "❌ No patient record found for your phone number."
        
        visit = get_visit_details(visit_id)
        if visit and visit.get('patient_id') == patient['patient_id']:
            return format_visit_details_for_whatsapp(visit)
        else:
            return "❌ Visit not found or you don't have access to this visit."
    
    return "❌ Unable to process your data request. Please try again."


# =============================================================================
# TEST/DEMO FUNCTIONS
# =============================================================================

if __name__ == "__main__":
    # Demo: Test with sample phone number
    print("=" * 60)
    print("🗄️ Database Utilities - Test Mode")
    print("=" * 60)
    
    # Test intent detection
    test_messages = [
        "show my data",
        "I want to see my records",
        "what medicines am I taking?",
        "visit id 1",
        "hello",  # Should not be a data query
        "1"  # Normal MCQ response - not a data query
    ]
    
    print("\n📝 Testing intent detection:")
    for msg in test_messages:
        result = detect_data_query_intent(msg)
        status = "✅ DATA QUERY" if result['is_data_query'] else "❌ Not a data query"
        print(f"  '{msg}' -> {status} (type: {result['query_type']})")
    
    print("\n📊 Testing database query (requires database connection):")
    test_phone = "+919876543210"
    data = get_patient_full_data(test_phone)
    
    if data:
        print(f"\n{format_patient_data_for_whatsapp(data)}")
    else:
        print(f"  No data found for {test_phone}")
        print("  (Make sure database is configured and has sample data)")
