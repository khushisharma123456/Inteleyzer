"""
Excel LLM Interpretation Service for the Pharmacovigilance system.
Uses LLM APIs (OpenAI/Azure OpenAI) to interpret Excel files in any format
and map them to the database schema.
"""
import os
import io
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd

# LLM providers - will use whichever is available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pv_backend.models import (
    db, ExperienceEvent, EventSource, CaseMaster, Reporter, User
)


class ExcelLLMService:
    """
    Service for interpreting Excel files using LLM and feeding data to the database.
    
    PV Context:
    - Adverse event reports come from many sources in various formats
    - LLM interprets unstructured/semi-structured Excel data
    - Maps to standard PV data model for case processing
    """
    
    # Target schema fields that LLM should extract
    TARGET_SCHEMA = {
        "drug_name": "Name of the drug/medication (required)",
        "drug_code": "Drug code like NDC, ATC code (optional)",
        "drug_batch": "Batch or lot number (optional)",
        "patient_identifier": "Patient ID, initials, or any identifier (for hashing)",
        "patient_age": "Patient age or date of birth (optional)",
        "patient_gender": "Patient gender (optional)",
        "indication": "Why the drug was prescribed/taken",
        "dosage": "Drug dosage and frequency",
        "route_of_administration": "How the drug was administered (oral, IV, topical, etc.)",
        "start_date": "When the drug was started (ISO format: YYYY-MM-DD)",
        "end_date": "When the drug was stopped (ISO format: YYYY-MM-DD, optional)",
        "event_date": "When the adverse event occurred (ISO format: YYYY-MM-DD)",
        "observed_events": "Description of symptoms, reactions, side effects observed",
        "outcome": "Outcome (recovered, ongoing, fatal, unknown)",
        "seriousness": "Is it serious? (death, hospitalization, disability, etc.)",
        "reporter_name": "Name of the person reporting (optional)",
        "reporter_type": "Type of reporter (doctor, pharmacist, patient, etc.)",
        "reporter_institution": "Hospital/clinic name (optional)",
        "additional_notes": "Any other relevant information"
    }
    
    def __init__(self, api_key: str = None, api_type: str = "openai", 
                 azure_endpoint: str = None, azure_deployment: str = None):
        """
        Initialize the Excel LLM Service.
        
        Args:
            api_key: OpenAI or Azure OpenAI API key
            api_type: 'openai' or 'azure'
            azure_endpoint: Azure OpenAI endpoint (for Azure)
            azure_deployment: Azure OpenAI deployment name (for Azure)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY') or os.environ.get('AZURE_OPENAI_API_KEY')
        self.api_type = api_type
        self.azure_endpoint = azure_endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT')
        self.azure_deployment = azure_deployment or os.environ.get('AZURE_OPENAI_DEPLOYMENT')
        
        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        # Initialize OpenAI client
        if self.api_type == "azure":
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=self.azure_endpoint
            )
            self.model = self.azure_deployment
        else:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = "gpt-4o"  # Best for structured data extraction
    
    def read_excel_to_text(self, file_path: str = None, file_content: bytes = None) -> Tuple[str, pd.DataFrame]:
        """
        Read Excel file and convert to text representation for LLM.
        
        Args:
            file_path: Path to Excel file
            file_content: Excel file content as bytes (for file uploads)
            
        Returns:
            Tuple of (text representation, DataFrame)
        """
        if file_content:
            df = pd.read_excel(io.BytesIO(file_content))
        elif file_path:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Either file_path or file_content must be provided")
        
        # Convert DataFrame to a text representation
        # Include column headers and sample data
        text_repr = f"Excel File Contents:\n"
        text_repr += f"Columns: {list(df.columns)}\n\n"
        text_repr += f"Number of rows: {len(df)}\n\n"
        
        # Include all data (or limit for very large files)
        max_rows = 100  # Limit to prevent token overflow
        if len(df) > max_rows:
            text_repr += f"(Showing first {max_rows} rows of {len(df)} total)\n\n"
        
        text_repr += df.head(max_rows).to_string()
        
        return text_repr, df
    
    def _build_extraction_prompt(self, excel_text: str) -> str:
        """Build the prompt for LLM data extraction."""
        
        schema_description = "\n".join([
            f"  - {field}: {desc}" 
            for field, desc in self.TARGET_SCHEMA.items()
        ])
        
        prompt = f"""You are a pharmacovigilance data extraction specialist. Your task is to interpret this Excel file 
containing adverse event or drug experience reports and extract the data into a standardized format.

The Excel file may have:
- Different column names than our standard schema
- Data in different formats (dates, names, etc.)
- Multiple rows, each representing a separate event/report
- Combined fields that need to be split
- Missing fields that should be left null

TARGET SCHEMA (extract these fields for each row):
{schema_description}

EXCEL DATA:
{excel_text}

INSTRUCTIONS:
1. Analyze the Excel structure and identify which columns map to which target fields
2. For each row, extract the relevant data and map it to our schema
3. Normalize dates to ISO format (YYYY-MM-DD) when possible
4. If a field can't be determined, use null
5. The 'observed_events' field is CRITICAL - include all symptom/reaction descriptions
6. Return a JSON array of objects, one per row

IMPORTANT:
- Be thorough in identifying adverse event descriptions
- Consider that column names might be in different languages or abbreviated
- Combine multiple columns if they describe the same target field
- drug_name and observed_events are the most important fields

Return ONLY valid JSON array, no additional text or explanation.
Example format:
[
  {{
    "drug_name": "Aspirin",
    "patient_identifier": "PT001",
    "observed_events": "Nausea and headache after first dose",
    ...
  }},
  ...
]
"""
        return prompt
    
    def interpret_excel(self, file_path: str = None, file_content: bytes = None) -> List[Dict[str, Any]]:
        """
        Use LLM to interpret Excel file and extract structured data.
        
        Args:
            file_path: Path to Excel file
            file_content: Excel file content as bytes
            
        Returns:
            List of dictionaries with extracted data mapped to target schema
        """
        # Read Excel file
        excel_text, df = self.read_excel_to_text(file_path=file_path, file_content=file_content)
        
        # Build prompt
        prompt = self._build_extraction_prompt(excel_text)
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a pharmacovigilance data extraction specialist. Extract adverse event data from Excel files into structured JSON format. Always return valid JSON arrays."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=4000,
            response_format={"type": "json_object"}  # Ensure JSON response
        )
        
        # Parse response
        result_text = response.choices[0].message.content
        
        try:
            result = json.loads(result_text)
            # Handle if LLM returns {"data": [...]} or just [...]
            if isinstance(result, dict) and "data" in result:
                return result["data"]
            elif isinstance(result, list):
                return result
            else:
                # Try to find array in response
                for key, value in result.items():
                    if isinstance(value, list):
                        return value
                return [result]  # Single record
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nResponse: {result_text[:500]}")
    
    def validate_extracted_data(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate extracted records and separate valid from invalid.
        
        Args:
            records: List of extracted data records
            
        Returns:
            Tuple of (valid_records, invalid_records_with_errors)
        """
        valid = []
        invalid = []
        
        for i, record in enumerate(records):
            errors = []
            
            # Check required fields
            if not record.get('drug_name'):
                errors.append("Missing required field: drug_name")
            
            if not record.get('observed_events'):
                errors.append("Missing required field: observed_events")
            
            # Validate dates if present
            for date_field in ['start_date', 'end_date', 'event_date']:
                if record.get(date_field):
                    try:
                        datetime.fromisoformat(str(record[date_field]).replace('/', '-'))
                    except ValueError:
                        errors.append(f"Invalid date format for {date_field}: {record[date_field]}")
            
            if errors:
                invalid.append({
                    'row_index': i,
                    'data': record,
                    'errors': errors
                })
            else:
                valid.append(record)
        
        return valid, invalid
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        date_str = str(date_str).strip()
        
        # Try common formats
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def create_experience_events(
        self, 
        records: List[Dict[str, Any]], 
        user: User,
        source: EventSource = EventSource.HOSPITAL
    ) -> List[Dict[str, Any]]:
        """
        Create ExperienceEvent records from extracted data.
        
        Args:
            records: List of validated records from LLM extraction
            user: User who is uploading the file
            source: Event source (default: HOSPITAL for bulk imports)
            
        Returns:
            List of result dictionaries with created event info
        """
        from pv_backend.services.normalization_service import NormalizationService
        from pv_backend.services.case_linking_service import CaseLinkingService
        from pv_backend.services.scoring_service import ScoringService
        
        results = []
        
        for record in records:
            try:
                # Hash patient identifier for privacy
                patient_id_raw = record.get('patient_identifier', '')
                if patient_id_raw:
                    patient_hash = hashlib.sha256(str(patient_id_raw).encode()).hexdigest()
                else:
                    patient_hash = None
                
                # Generate idempotency key
                idempotency_key = hashlib.sha256(
                    f"{source.value}:{user.id}:{record.get('drug_name', '')}:"
                    f"{patient_hash or ''}:{record.get('event_date', '')}:"
                    f"{record.get('observed_events', '')[:50]}".encode()
                ).hexdigest()
                
                # Check for duplicate
                existing_event = ExperienceEvent.query.filter_by(
                    idempotency_key=idempotency_key
                ).first()
                
                if existing_event:
                    results.append({
                        'success': True,
                        'is_duplicate': True,
                        'event_id': existing_event.id,
                        'case_id': existing_event.case_id,
                        'drug_name': record.get('drug_name'),
                        'message': 'Duplicate record - already exists'
                    })
                    continue
                
                # Create experience event
                event = ExperienceEvent(
                    idempotency_key=idempotency_key,
                    source=source,
                    submitted_by_user_id=user.id,
                    drug_name=record.get('drug_name'),
                    drug_code=record.get('drug_code'),
                    drug_batch=record.get('drug_batch'),
                    patient_identifier_hash=patient_hash,
                    indication=record.get('indication'),
                    dosage=record.get('dosage'),
                    route_of_administration=record.get('route_of_administration'),
                    start_date=self._parse_date(record.get('start_date')),
                    end_date=self._parse_date(record.get('end_date')),
                    event_date=self._parse_date(record.get('event_date')) or datetime.utcnow(),
                    observed_events=record.get('observed_events'),
                    outcome=record.get('outcome'),
                    raw_payload=record  # Store original extracted data
                )
                
                db.session.add(event)
                db.session.flush()
                
                # Process through normalization pipeline
                normalized = NormalizationService.normalize_event(event)
                
                # Link to case
                case = CaseLinkingService.link_event_to_case(event)
                
                # Update case score
                if case:
                    ScoringService.update_case_score(case)
                
                results.append({
                    'success': True,
                    'is_duplicate': False,
                    'event_id': event.id,
                    'case_id': case.id if case else None,
                    'case_number': case.case_number if case else None,
                    'drug_name': record.get('drug_name'),
                    'message': 'Successfully created'
                })
                
            except Exception as e:
                db.session.rollback()
                results.append({
                    'success': False,
                    'error': str(e),
                    'drug_name': record.get('drug_name'),
                    'message': f'Failed to create event: {str(e)}'
                })
        
        # Commit all successful records
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise
        
        return results
    
    def process_excel_file(
        self,
        user: User,
        file_path: str = None,
        file_content: bytes = None,
        source: EventSource = EventSource.HOSPITAL
    ) -> Dict[str, Any]:
        """
        Complete pipeline: Read Excel → LLM Interpret → Validate → Create Events.
        
        Args:
            user: User performing the upload
            file_path: Path to Excel file
            file_content: Excel file content as bytes
            source: Event source
            
        Returns:
            Processing result summary
        """
        start_time = datetime.utcnow()
        
        # Step 1: LLM Interpretation
        extracted_records = self.interpret_excel(
            file_path=file_path, 
            file_content=file_content
        )
        
        # Step 2: Validation
        valid_records, invalid_records = self.validate_extracted_data(extracted_records)
        
        # Step 3: Create events for valid records
        creation_results = []
        if valid_records:
            creation_results = self.create_experience_events(
                records=valid_records,
                user=user,
                source=source
            )
        
        # Compile summary
        successful = [r for r in creation_results if r.get('success') and not r.get('is_duplicate')]
        duplicates = [r for r in creation_results if r.get('is_duplicate')]
        failed = [r for r in creation_results if not r.get('success')]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'success': True,
            'processing_time_seconds': processing_time,
            'summary': {
                'total_rows_extracted': len(extracted_records),
                'valid_records': len(valid_records),
                'invalid_records': len(invalid_records),
                'successfully_created': len(successful),
                'duplicates_skipped': len(duplicates),
                'failed_to_create': len(failed)
            },
            'created_events': successful,
            'duplicates': duplicates,
            'validation_errors': invalid_records,
            'creation_errors': failed
        }


class ExcelLLMServiceFactory:
    """Factory for creating ExcelLLMService with proper configuration."""
    
    @staticmethod
    def create_from_env() -> ExcelLLMService:
        """Create service using environment variables."""
        api_type = os.environ.get('LLM_API_TYPE', 'openai')
        
        return ExcelLLMService(
            api_type=api_type,
            api_key=os.environ.get('OPENAI_API_KEY') or os.environ.get('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            azure_deployment=os.environ.get('AZURE_OPENAI_DEPLOYMENT')
        )
    
    @staticmethod
    def create_openai(api_key: str, model: str = "gpt-4o") -> ExcelLLMService:
        """Create service using OpenAI API."""
        service = ExcelLLMService(api_key=api_key, api_type="openai")
        service.model = model
        return service
    
    @staticmethod
    def create_azure(api_key: str, endpoint: str, deployment: str) -> ExcelLLMService:
        """Create service using Azure OpenAI API."""
        return ExcelLLMService(
            api_key=api_key,
            api_type="azure",
            azure_endpoint=endpoint,
            azure_deployment=deployment
        )
