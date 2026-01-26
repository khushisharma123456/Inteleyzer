# Excel LLM Interpretation Feature

## Overview

This feature allows the Pharmacovigilance system to accept Excel files in **any format** and use an LLM (Large Language Model) to intelligently interpret and map the data to the database schema. This eliminates the need for predefined Excel templates and handles various report formats from different sources.

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Excel Upload   │────>│ LLM Interprets   │────>│   Validation     │────>│ Create Events    │
│   (Any Format)   │     │   & Maps Data    │     │   & Extraction   │     │ in Database      │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

1. **Upload**: User uploads an Excel file (`.xlsx`, `.xls`, `.xlsm`)
2. **Interpretation**: LLM analyzes column headers and data to understand the structure
3. **Mapping**: LLM maps each column to the target PV data schema
4. **Validation**: Extracted data is validated for required fields and formats
5. **Creation**: Valid records are processed through the normal PV pipeline (normalization → case linking → scoring)

## Configuration

### Environment Variables

Set one of the following configurations:

#### Option 1: OpenAI API
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
export LLM_API_TYPE="openai"  # Optional, defaults to "openai"
```

#### Option 2: Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"  # Your deployment name
export LLM_API_TYPE="azure"
```

### Install Dependencies
```bash
pip install openai pandas openpyxl xlrd
```

## API Endpoints

### 1. Upload and Process Excel
```http
POST /api/excel/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <excel_file>
source: hospital  # optional: doctor, hospital, pharmacy
```

**Response:**
```json
{
  "success": true,
  "processing_time_seconds": 5.2,
  "summary": {
    "total_rows_extracted": 50,
    "valid_records": 48,
    "invalid_records": 2,
    "successfully_created": 45,
    "duplicates_skipped": 3,
    "failed_to_create": 0
  },
  "created_events": [...],
  "duplicates": [...],
  "validation_errors": [...],
  "creation_errors": [...]
}
```

### 2. Preview Interpretation (Dry Run)
```http
POST /api/excel/preview
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <excel_file>
```

**Response:**
```json
{
  "success": true,
  "preview": true,
  "total_records": 50,
  "valid_count": 48,
  "invalid_count": 2,
  "extracted_data": [...],
  "valid_records": [...],
  "validation_errors": [...],
  "target_schema": {...}
}
```

### 3. Get Target Schema
```http
GET /api/excel/schema
Authorization: Bearer <token>
```

### 4. Download Template
```http
GET /api/excel/template
Authorization: Bearer <token>
```

### 5. Check Service Status
```http
GET /api/excel/status
Authorization: Bearer <token>
```

## Target Schema

The LLM maps Excel data to these fields:

| Field | Description | Required |
|-------|-------------|----------|
| `drug_name` | Name of the drug/medication | ✅ Yes |
| `drug_code` | Drug code (NDC, ATC, etc.) | No |
| `drug_batch` | Batch/lot number | No |
| `patient_identifier` | Patient ID (will be hashed) | No |
| `patient_age` | Patient age or DOB | No |
| `patient_gender` | Patient gender | No |
| `indication` | Why drug was prescribed | No |
| `dosage` | Drug dosage and frequency | No |
| `route_of_administration` | Oral, IV, topical, etc. | No |
| `start_date` | When drug was started | No |
| `end_date` | When drug was stopped | No |
| `event_date` | When adverse event occurred | No |
| `observed_events` | Symptoms, reactions observed | ✅ Yes |
| `outcome` | Recovered, ongoing, fatal, etc. | No |
| `seriousness` | Serious criteria (death, etc.) | No |
| `reporter_name` | Person reporting | No |
| `reporter_type` | Doctor, pharmacist, patient | No |
| `reporter_institution` | Hospital/clinic name | No |
| `additional_notes` | Other information | No |

## Example Excel Formats Supported

The LLM can interpret various formats:

### Format 1: Standard Headers
| Drug | Patient | Reaction | Date |
|------|---------|----------|------|
| Aspirin | PT001 | Nausea, vomiting | 2024-01-15 |

### Format 2: Clinical Format
| Medication Name | MRN | AE Description | Onset Date | Severity |
|-----------------|-----|----------------|------------|----------|
| Metformin 500mg | 12345 | Abdominal pain | 15/01/2024 | Moderate |

### Format 3: Pharmacy Report
| Product | Customer ID | Complaint | Report Date | Batch |
|---------|------------|-----------|-------------|-------|
| Ibuprofen Tablets | C789 | Allergic rash | 2024-01-20 | LOT456 |

### Format 4: Multi-language Headers
| 薬品名 | 患者ID | 副作用 | 発生日 |
|-------|--------|-------|-------|
| アスピリン | PT002 | 頭痛 | 2024-01-18 |

## Usage Examples

### Python
```python
import requests

# Upload Excel file
with open('adverse_events.xlsx', 'rb') as f:
    response = requests.post(
        'http://localhost:5001/api/excel/upload',
        headers={'Authorization': f'Bearer {token}'},
        files={'file': f},
        data={'source': 'hospital'}
    )

result = response.json()
print(f"Created {result['summary']['successfully_created']} events")
```

### cURL
```bash
curl -X POST http://localhost:5001/api/excel/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@adverse_events.xlsx" \
  -F "source=hospital"
```

### JavaScript (Frontend)
```javascript
const formData = new FormData();
formData.append('file', excelFile);
formData.append('source', 'hospital');

const response = await fetch('/api/excel/upload', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`
    },
    body: formData
});

const result = await response.json();
console.log(`Processed ${result.summary.total_rows_extracted} rows`);
```

## Best Practices

1. **Preview First**: Use `/api/excel/preview` to review interpretation before committing
2. **Validate Results**: Check `validation_errors` in the response for issues
3. **Handle Duplicates**: The system automatically detects and skips duplicates
4. **Monitor Costs**: LLM API calls incur costs; batch uploads when possible
5. **Date Formats**: While LLM handles various formats, ISO dates (YYYY-MM-DD) work best

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "Service not configured" | Missing API key | Set OPENAI_API_KEY or Azure vars |
| "Invalid file type" | Wrong file extension | Use .xlsx, .xls, or .xlsm |
| "Missing required field" | No drug_name or observed_events | Ensure Excel has these columns |
| "LLM returned invalid JSON" | LLM parsing error | Retry or simplify Excel structure |

## Security Notes

- Patient identifiers are **automatically hashed** before storage
- Original Excel data is stored in `raw_payload` for audit purposes
- All uploads are logged in the audit trail
- API requires authentication (JWT token)
