# Pharmacy Reports Module - Complete Implementation Guide

## Overview

The Pharmacy Reports Module is a three-section workflow system for submitting pharmacy safety data in controlled formats. It enforces strict schema validation, supports multiple data entry modes, and maintains audit trails for regulatory compliance.

---

## Architecture

### Three-Section Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ SECTION 1: Data Type Selector (Mandatory First Step)        │
│ - Anonymous Data (DEFAULT)                                  │
│ - Data with Identity (Restricted, Consent-based)            │
│ - Aggregated / Disease Analysis (Excel-only)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ SECTION 2: Data Entry Mode Selector                         │
│ - Manual Entry (Form-based)                                 │
│ - Excel Upload (Bulk submission)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ SECTION 3: Submission Workspace                             │
│ - Form rendering or Excel validation                        │
│ - Preview and confirmation                                  │
│ - Final submission with compliance checkbox                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Types & Schemas

### 1. Anonymous Data (DEFAULT)

**Purpose**: No personal identifiers. Safe for all pharmacies. Highest priority for analytics.

**Required Fields**:
- Drug Name
- Dosage Form
- Date of Dispensing
- Reaction Category
- Severity (mild, moderate, severe)
- Age Group (pediatric, adolescent, adult, elderly, unknown)

**Optional Fields**:
- Batch / Lot Number
- Reaction Outcome (recovered, recovering, not_recovered, fatal, unknown)
- Gender (male, female, other, not_specified)
- Additional Notes

**Use Case**: General adverse event reporting without patient identification.

---

### 2. Data with Identity (RESTRICTED)

**Purpose**: Limited identifiers only. Requires consent verification.

**Allowed Identifiers**:
- Internal Case ID
- Treating Hospital / Doctor Reference
- Treating Doctor Name

**❌ NOT Allowed**:
- Direct patient contact fields (phone, email, address)
- Patient name
- Patient ID numbers

**Required Fields** (in addition to Anonymous):
- Consent Verified (checkbox)

**Optional Fields** (in addition to Anonymous):
- Internal Case ID
- Treating Hospital Reference
- Treating Doctor Name
- Consent Date

**Use Case**: Follow-up investigations, case linkage, hospital coordination.

**Warning Message**:
> ⚠️ Important Notice
> Submit identifiable data only if consent is available. Allowed identifiers: Internal case ID, Treating hospital/doctor reference. Direct patient contact fields are NOT allowed.

---

### 3. Aggregated / Disease Analysis

**Purpose**: Summary counts only. No individual case rows. Excel-only.

**Required Fields**:
- Drug Name
- Total Dispensed
- Total Reactions Reported
- Mild Count
- Moderate Count
- Severe Count
- Reporting Period Start
- Reporting Period End

**Optional Fields**:
- Analysis Notes

**Use Case**: Trend analysis, pattern detection, regulatory reporting.

**Restrictions**:
- Excel upload only (no manual entry)
- Summary data only (no individual records)

---

## Entry Modes

### Manual Entry

**Workflow**:
1. User selects report type
2. Form fields render based on schema
3. User fills in one record at a time
4. "Add Another Record" button allows multiple entries
5. Validation on submit
6. Submission summary with confirmation checkbox

**Features**:
- Real-time field validation
- Required field indicators (*)
- Helpful placeholders
- Remove button for extra records
- Clear error messages

---

### Excel Upload

**Workflow**:
1. User selects report type
2. Download template button provided
3. User uploads Excel file
4. Schema validation (column names must match exactly)
5. Preview of first 10 rows
6. Confirmation and submit

**Template Requirements**:
- Column names must match schema exactly
- Order does not matter
- Case-insensitive matching allowed
- Extra columns ❌ NOT allowed
- Missing columns ❌ NOT allowed

**Validation Rules**:
```javascript
// Column validation
- All required columns must exist
- Column names are case-insensitive
- Extra columns are rejected
- Missing columns are rejected
```

**Error Handling**:
If validation fails, show white prompt card (non-blocking):
```
❌ Invalid Excel Format
The uploaded file does not contain the required columns.

Required Columns:
- Drug Name
- Batch Number
- Dosage Form
- Reaction Category
- Severity
- Age Group

Please update your file or download the correct template.
```

---

## Frontend Implementation

### File: `templates/pharmacy/reports.html`

**Structure**:
- Three vertical sections (not tabs)
- Step indicators (1, 2, 3)
- Responsive grid layout
- Clear visual hierarchy

**Key Components**:
1. **Data Type Selector**: Radio buttons with cards
2. **Entry Mode Selector**: Radio buttons with cards
3. **Form Fields Container**: Dynamic form rendering
4. **Excel Upload Area**: Drag-and-drop support
5. **Validation Prompt**: Non-blocking error/success messages
6. **Preview Table**: Shows first 10 rows of Excel data
7. **Submission Summary**: Confirmation before final submit

---

### File: `static/js/pharmacy-reports.js`

**State Management**:
```javascript
const reportState = {
    reportType: 'anonymous',      // anonymous, identified, aggregated
    entryMode: 'manual',          // manual, excel
    schema: [],                   // Current schema
    records: [],                  // Collected records
    excelData: null,              // Excel data after validation
    columnMapping: null           // Column mapping for Excel
};
```

**Key Functions**:

#### Initialization
- `initializeEventListeners()`: Set up all event handlers
- `renderFormFields()`: Render form based on schema
- `updateStepIndicators()`: Update step badges

#### Form Management
- `addFormRow()`: Add new record form
- `removeFormRow(rowIndex)`: Remove record form
- `renderFormField(field, rowIndex)`: Render single field
- `collectFormData()`: Collect all form data
- `validateFormData(records)`: Validate collected data

#### Excel Handling
- `downloadTemplate()`: Generate and download Excel template
- `handleFileSelect(event)`: Process uploaded file
- `showExcelPreview(rows)`: Display preview table

#### Submission
- `updateSummary(recordCount)`: Update submission summary
- `submitReport()`: Submit data to backend
- `resetForm()`: Clear all form data

---

## Backend Implementation

### File: `pv_backend/routes/pharmacy_report_routes.py`

**Endpoints**:

#### 1. Submit Report
```
POST /api/pharmacy/reports/submit
Content-Type: application/json

{
    "report_type": "anonymous|identified|aggregated",
    "entry_mode": "manual|excel",
    "records": [
        {
            "drug_name": "Aspirin 500mg",
            "batch_lot_number": "LOT-2024-001",
            "dosage_form": "Tablet",
            "date_of_dispensing": "2024-01-15",
            "reaction_category": "Gastrointestinal",
            "severity": "mild",
            "reaction_outcome": "recovered",
            "age_group": "adult",
            "gender": "female",
            "additional_notes": "..."
        }
    ]
}

Response (201):
{
    "success": true,
    "submission_id": "SUB-20240128193415",
    "message": "Successfully submitted 5 records",
    "record_count": 5
}
```

#### 2. Validate Excel
```
POST /api/pharmacy/reports/validate-excel
Content-Type: multipart/form-data

file: <Excel file>
report_type: anonymous|identified|aggregated

Response (200):
{
    "success": true,
    "total_rows": 25,
    "preview_rows": [
        {
            "Drug Name": "Aspirin 500mg",
            "Batch / Lot Number": "LOT-2024-001",
            ...
        }
    ],
    "column_mapping": {...}
}

Response (400) - Validation Failed:
{
    "success": false,
    "error": "Missing required columns: Severity, Age Group",
    "required_columns": ["Drug Name", "Batch Number", ...]
}
```

#### 3. Get Submission History
```
GET /api/pharmacy/reports/history?limit=50&offset=0

Response (200):
{
    "success": true,
    "reports": [
        {
            "id": 1,
            "report_type": "anonymous",
            "drug_name": "Aspirin 500mg",
            "created_at": "2024-01-28T19:34:15",
            "status": "submitted"
        }
    ],
    "total": 25
}
```

#### 4. Get Report Detail
```
GET /api/pharmacy/reports/<report_id>

Response (200):
{
    "success": true,
    "report": {
        "id": 1,
        "report_type": "anonymous",
        "drug_name": "Aspirin 500mg",
        "reaction_description": "Gastrointestinal upset",
        "severity": "mild",
        "created_at": "2024-01-28T19:34:15"
    }
}
```

#### 5. Get Compliance Score
```
GET /api/pharmacy/reports/compliance-score

Response (200):
{
    "success": true,
    "compliance_score": 85,
    "status": "Compliant",
    "status_color": "green",
    "last_updated": "2024-01-28T19:34:15"
}
```

---

## Database Models

### PharmacyReport (Base Model)
```python
class PharmacyReport(db.Model):
    id: Integer (PK)
    report_type: Enum (ANONYMOUS, IDENTIFIED, AGGREGATED)
    pharmacy_id: Integer (FK to User)
    drug_name: String
    drug_batch_number: String
    reaction_description: Text
    reaction_severity: Enum (MILD, MODERATE, SEVERE, LIFE_THREATENING)
    reaction_outcome: Enum (RECOVERED, RECOVERING, NOT_RECOVERED, FATAL, UNKNOWN)
    created_at: DateTime
    updated_at: DateTime
    is_deleted: Boolean
```

### AnonymousReport (Inherits PharmacyReport)
```python
class AnonymousReport(PharmacyReport):
    age_group: Enum (PEDIATRIC, ADOLESCENT, ADULT, ELDERLY, UNKNOWN)
    gender: String
    concomitant_medications: Text
    medical_history: Text
```

### IdentifiedReport (Inherits PharmacyReport)
```python
class IdentifiedReport(PharmacyReport):
    patient_name: String
    patient_age: Integer
    patient_gender: String
    patient_phone: String
    patient_email: String
    prescribing_doctor_name: String
    prescribing_doctor_phone: String
    hospital_name: String
    indication: Text
    concomitant_medications: Text
    medical_history: Text
    allergies: Text
    follow_up_required: Boolean
    follow_up_date: Date
    follow_up_notes: Text
```

### AggregatedReport (Inherits PharmacyReport)
```python
class AggregatedReport(PharmacyReport):
    report_count: Integer
    time_period_start: Date
    time_period_end: Date
    age_group_distribution: JSON
    gender_distribution: JSON
    common_reactions: JSON
    severity_distribution: JSON
    outcome_distribution: JSON
    trend_direction: String (increasing, decreasing, stable)
    trend_percentage: Float
```

---

## Compliance Scoring

### Scoring Algorithm

**Base Score**: 100 points

**Positive Behaviors** (+points):
- On-time ADR submission: +5
- Alert acknowledged within 24 hrs: +3
- Severe ADR reported within SLA: +10
- Consistent reporting frequency: +2

**Negative Behaviors** (-points):
- Late submission: -5
- Ignored alert: -10
- Invalid Excel upload: -2
- Missing mandatory fields: -3
- Long gaps in reporting: -5

### Compliance Status Labels

```
🟢 Compliant (80–100)
   - All submissions on time
   - Alerts acknowledged promptly
   - Data quality excellent

🟡 Attention Required (60–79)
   - Some late submissions
   - Occasional schema violations
   - Needs improvement

🔴 Non-compliant (<60)
   - Frequent late submissions
   - Repeated schema violations
   - Requires intervention
```

### Endpoint
```
GET /api/pharmacy/reports/compliance-score

Response:
{
    "success": true,
    "compliance_score": 85,
    "status": "Compliant",
    "status_color": "green",
    "last_updated": "2024-01-28T19:34:15"
}
```

---

## UX & Compliance Rules (NON-NEGOTIABLE)

✅ **DO**:
- Manual form = schema authority
- Excel must follow manual schema exactly
- Show clear educational prompts
- Validate before accepting data
- Maintain audit trail
- Reject invalid data with guidance
- Show step-by-step workflow
- Provide template download

❌ **DON'T**:
- Silent failures
- Auto-mapping columns
- Partial imports
- Patient contact fields in identified data
- Editing after submission
- Ambiguous error messages
- Mixing data types in one submission

---

## Testing Checklist

### Manual Entry
- [ ] Form renders correctly for each report type
- [ ] Required fields are marked with *
- [ ] Add/Remove record buttons work
- [ ] Validation catches missing required fields
- [ ] Submission summary shows correct data
- [ ] Confirmation checkbox enables submit button

### Excel Upload
- [ ] Template download works
- [ ] Template has correct columns
- [ ] File upload accepts .xlsx and .xls
- [ ] Drag-and-drop works
- [ ] Schema validation rejects missing columns
- [ ] Schema validation rejects extra columns
- [ ] Preview shows first 10 rows
- [ ] Submission works with valid file

### Data Types
- [ ] Anonymous: No identifiers shown
- [ ] Identified: Warning message displays
- [ ] Identified: Consent checkbox required
- [ ] Aggregated: Excel-only (manual disabled)

### Compliance
- [ ] Compliance score calculates correctly
- [ ] Status labels display correctly
- [ ] Audit log records submissions
- [ ] Submission history shows all records

---

## Security Considerations

1. **Authentication**: All endpoints require session authentication
2. **Authorization**: Users can only access their own pharmacy data
3. **Data Validation**: All inputs validated server-side
4. **Schema Enforcement**: No deviations from defined schema
5. **Audit Trail**: All submissions logged with timestamp and user ID
6. **Soft Delete**: Records marked as deleted, not removed
7. **CORS**: Restricted to allowed origins

---

## Future Enhancements

1. **Batch Processing**: Handle large Excel files asynchronously
2. **Data Mapping**: Allow column remapping for Excel uploads
3. **Conditional Fields**: Show/hide fields based on selections
4. **Real-time Validation**: Validate as user types
5. **Duplicate Detection**: Warn about potential duplicate submissions
6. **Export Reports**: Generate PDF/Excel reports from submissions
7. **Analytics Dashboard**: Visualize submission trends
8. **Integration**: Connect to regulatory reporting systems

---

## Support & Documentation

- **Frontend**: `templates/pharmacy/reports.html`
- **JavaScript**: `static/js/pharmacy-reports.js`
- **Backend**: `pv_backend/routes/pharmacy_report_routes.py`
- **Models**: `pv_backend/models/pharmacy_report.py`
- **Audit**: `pv_backend/services/audit_service.py`

---

## Version History

- **v1.0** (2024-01-28): Initial implementation
  - Three-section workflow
  - Manual entry and Excel upload
  - Schema validation
  - Compliance scoring
  - Audit logging
