"""
Excel Upload API routes for the Pharmacovigilance system.
Handles Excel file uploads with LLM-based interpretation.
"""
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, g
from werkzeug.utils import secure_filename

from pv_backend.models import db, User, EventSource, AuditLog, AuditAction
from pv_backend.auth import token_required, roles_required
from pv_backend.services.excel_llm_service import ExcelLLMService, ExcelLLMServiceFactory
from pv_backend.services.audit_service import AuditService


excel_upload_bp = Blueprint('excel_upload', __name__, url_prefix='/api/excel')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'xlsm'}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_llm_service() -> ExcelLLMService:
    """Get or create LLM service instance."""
    try:
        return ExcelLLMServiceFactory.create_from_env()
    except (ValueError, ImportError) as e:
        raise RuntimeError(f"LLM service not configured: {str(e)}")


@excel_upload_bp.route('/upload', methods=['POST'])
@token_required
def upload_excel():
    """
    Upload and process an Excel file with adverse event data.
    
    PV Context:
    - Accepts Excel files in ANY format
    - LLM interprets and maps data to standard schema
    - Creates experience events for each valid row
    - Returns detailed processing results
    
    Request:
        - file: Excel file (multipart/form-data)
        - source: Event source (optional, default: 'hospital')
    
    Response:
        - Processing summary with created events, errors, etc.
    """
    user = g.current_user
    
    # Check for file in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided',
            'message': 'Please upload an Excel file'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename',
            'message': 'Please select a file to upload'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type',
            'message': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    # Get source from request
    source_str = request.form.get('source', 'hospital')
    try:
        source = EventSource(source_str.lower())
    except ValueError:
        source = EventSource.HOSPITAL
    
    try:
        # Get LLM service
        llm_service = get_llm_service()
        
        # Read file content
        file_content = file.read()
        
        # Process the Excel file
        result = llm_service.process_excel_file(
            user=user,
            file_content=file_content,
            source=source
        )
        
        # Audit log
        AuditService.log_create(
            user=user,
            entity_type='ExcelUpload',
            entity_id=None,
            details={
                'filename': secure_filename(file.filename),
                'source': source.value,
                'summary': result.get('summary')
            }
        )
        
        return jsonify(result), 200
        
    except RuntimeError as e:
        return jsonify({
            'success': False,
            'error': 'Service not configured',
            'message': str(e)
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Processing failed',
            'message': str(e)
        }), 500


@excel_upload_bp.route('/preview', methods=['POST'])
@token_required
def preview_excel():
    """
    Preview Excel file interpretation without creating events.
    
    PV Context:
    - Allows users to review LLM interpretation before committing
    - Shows mapped data and validation errors
    - No database changes are made
    
    Request:
        - file: Excel file (multipart/form-data)
    
    Response:
        - Extracted and validated data preview
    """
    user = g.current_user
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type',
            'message': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        llm_service = get_llm_service()
        
        file_content = file.read()
        
        # Step 1: LLM Interpretation
        extracted_records = llm_service.interpret_excel(file_content=file_content)
        
        # Step 2: Validation (but don't create events)
        valid_records, invalid_records = llm_service.validate_extracted_data(extracted_records)
        
        return jsonify({
            'success': True,
            'preview': True,
            'filename': secure_filename(file.filename),
            'total_records': len(extracted_records),
            'valid_count': len(valid_records),
            'invalid_count': len(invalid_records),
            'extracted_data': extracted_records,
            'valid_records': valid_records,
            'validation_errors': invalid_records,
            'target_schema': ExcelLLMService.TARGET_SCHEMA
        }), 200
        
    except RuntimeError as e:
        return jsonify({
            'success': False,
            'error': 'Service not configured',
            'message': str(e)
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Preview failed',
            'message': str(e)
        }), 500


@excel_upload_bp.route('/schema', methods=['GET'])
@token_required
def get_target_schema():
    """
    Get the target schema that LLM will map Excel data to.
    
    PV Context:
    - Helps users understand what fields will be extracted
    - Can be used to create properly formatted Excel templates
    """
    return jsonify({
        'success': True,
        'schema': ExcelLLMService.TARGET_SCHEMA,
        'required_fields': ['drug_name', 'observed_events'],
        'date_fields': ['start_date', 'end_date', 'event_date'],
        'date_format': 'YYYY-MM-DD (ISO format recommended)',
        'notes': [
            'The LLM can interpret Excel files in any format',
            'Column names do not need to match the schema exactly',
            'The LLM will attempt to map your columns to the target fields',
            'drug_name and observed_events are required fields',
            'Dates should be in a recognizable format (YYYY-MM-DD preferred)'
        ]
    }), 200


@excel_upload_bp.route('/template', methods=['GET'])
@token_required
def download_template():
    """
    Get a recommended Excel template structure.
    
    PV Context:
    - Provides optimal column structure for best LLM interpretation
    - Users can use any format, but this template ensures best results
    """
    import io
    import pandas as pd
    from flask import send_file
    
    # Create template DataFrame
    template_data = {
        'Drug Name': ['Example Drug A', 'Example Drug B'],
        'Drug Code (NDC/ATC)': ['12345-678-90', '98765-432-10'],
        'Batch Number': ['LOT001', 'LOT002'],
        'Patient ID': ['PT001', 'PT002'],
        'Patient Age': [45, 62],
        'Patient Gender': ['Male', 'Female'],
        'Indication': ['Hypertension', 'Diabetes Type 2'],
        'Dosage': ['10mg once daily', '500mg twice daily'],
        'Route': ['Oral', 'Oral'],
        'Start Date': ['2024-01-15', '2024-02-01'],
        'End Date': ['2024-03-15', ''],
        'Event Date': ['2024-02-20', '2024-02-28'],
        'Observed Events/Reactions': [
            'Mild nausea and dizziness reported 1 week after starting treatment',
            'Skin rash on arms and legs, onset 3 days after dose increase'
        ],
        'Outcome': ['Recovered', 'Ongoing'],
        'Seriousness': ['Non-serious', 'Serious - required hospitalization'],
        'Reporter Name': ['Dr. Smith', 'Dr. Johnson'],
        'Reporter Institution': ['City Hospital', 'General Clinic'],
        'Additional Notes': ['Patient continued treatment', 'Treatment discontinued']
    }
    
    df = pd.DataFrame(template_data)
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Adverse Events')
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='adverse_event_template.xlsx'
    )


@excel_upload_bp.route('/status', methods=['GET'])
@token_required
def check_service_status():
    """
    Check if the LLM service is properly configured.
    
    PV Context:
    - Helps diagnose configuration issues
    - Returns service availability status
    """
    status = {
        'openai_key_configured': bool(os.environ.get('OPENAI_API_KEY')),
        'azure_key_configured': bool(os.environ.get('AZURE_OPENAI_API_KEY')),
        'azure_endpoint_configured': bool(os.environ.get('AZURE_OPENAI_ENDPOINT')),
        'azure_deployment_configured': bool(os.environ.get('AZURE_OPENAI_DEPLOYMENT')),
        'llm_api_type': os.environ.get('LLM_API_TYPE', 'openai')
    }
    
    # Check if service can be created
    try:
        get_llm_service()
        status['service_available'] = True
        status['message'] = 'LLM service is configured and available'
    except Exception as e:
        status['service_available'] = False
        status['message'] = str(e)
    
    return jsonify(status), 200
