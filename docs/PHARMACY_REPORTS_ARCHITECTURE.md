# Pharmacy Reports Module - Technical Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHARMACY DASHBOARD                        │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              REPORTS PAGE (reports.html)                 │   │
│  │                                                           │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ SECTION 1: Data Type Selector                   │    │   │
│  │  │ - Anonymous Data (default)                      │    │   │
│  │  │ - Data with Identity (restricted)               │    │   │
│  │  │ - Aggregated / Disease Analysis                 │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                        ↓                                  │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ SECTION 2: Entry Mode Selector                  │    │   │
│  │  │ - Manual Entry                                  │    │   │
│  │  │ - Excel Upload                                  │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                        ↓                                  │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ SECTION 3: Submission Workspace                 │    │   │
│  │  │ - Form Rendering / Excel Upload                 │    │   │
│  │  │ - Validation & Preview                          │    │   │
│  │  │ - Submission Summary & Confirmation             │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  JavaScript: pharmacy-reports.js                                 │
│  - State Management                                              │
│  - Event Handling                                                │
│  - Form Rendering                                                │
│  - Excel Processing                                              │
│  - API Communication                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND API LAYER                           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ pharmacy_report_routes.py                                │   │
│  │                                                           │   │
│  │ POST /api/pharmacy/reports/submit                        │   │
│  │ ├─ Validate schema                                       │   │
│  │ ├─ Create records                                        │   │
│  │ ├─ Log submission                                        │   │
│  │ └─ Return submission ID                                  │   │
│  │                                                           │   │
│  │ POST /api/pharmacy/reports/validate-excel               │   │
│  │ ├─ Read Excel file                                       │   │
│  │ ├─ Check column names                                    │   │
│  │ ├─ Generate preview                                      │   │
│  │ └─ Return validation result                              │   │
│  │                                                           │   │
│  │ GET /api/pharmacy/reports/history                        │   │
│  │ ├─ Query submissions                                     │   │
│  │ ├─ Apply pagination                                      │   │
│  │ └─ Return history                                        │   │
│  │                                                           │   │
│  │ GET /api/pharmacy/reports/compliance-score              │   │
│  │ ├─ Calculate score                                       │   │
│  │ ├─ Determine status                                      │   │
│  │ └─ Return score & status                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATABASE LAYER (SQLAlchemy)                   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ pharmacy_report.py (Models)                              │   │
│  │                                                           │   │
│  │ PharmacyReport (Base)                                    │   │
│  │ ├─ id, report_type, pharmacy_id                          │   │
│  │ ├─ drug_name, reaction_description                       │   │
│  │ ├─ severity, outcome                                     │   │
│  │ └─ created_at, updated_at                                │   │
│  │                                                           │   │
│  │ AnonymousReport (Inherits)                               │   │
│  │ ├─ age_group, gender                                     │   │
│  │ └─ concomitant_medications, medical_history              │   │
│  │                                                           │   │
│  │ IdentifiedReport (Inherits)                              │   │
│  │ ├─ patient_name, patient_age                             │   │
│  │ ├─ hospital_name, doctor_name                            │   │
│  │ └─ follow_up_required, follow_up_date                    │   │
│  │                                                           │   │
│  │ AggregatedReport (Inherits)                              │   │
│  │ ├─ report_count, time_period                             │   │
│  │ ├─ severity_distribution, outcome_distribution           │   │
│  │ └─ trend_direction, trend_percentage                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Database Tables                                          │   │
│  │ - pharmacy_reports (base table)                          │   │
│  │ - anonymous_reports (inherits)                           │   │
│  │ - identified_reports (inherits)                          │   │
│  │ - aggregated_reports (inherits)                          │   │
│  │ - submission_logs (audit trail)                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

### Manual Entry Flow

```
User Input (Form)
    ↓
Frontend Validation (pharmacy-reports.js)
    ├─ Check required fields
    ├─ Validate data types
    └─ Validate formats
    ↓
Collect Form Data
    ├─ Iterate through form rows
    ├─ Extract field values
    └─ Build records array
    ↓
API Call: POST /api/pharmacy/reports/submit
    ├─ report_type: "anonymous|identified|aggregated"
    ├─ entry_mode: "manual"
    └─ records: [...]
    ↓
Backend Validation (pharmacy_report_routes.py)
    ├─ Validate report type
    ├─ Validate schema
    ├─ Check required fields
    └─ Validate data types
    ↓
Create Database Records
    ├─ Instantiate model objects
    ├─ Set field values
    └─ Add to session
    ↓
Commit to Database
    ├─ Insert records
    ├─ Generate IDs
    └─ Update timestamps
    ↓
Log Submission (audit_service.py)
    ├─ Record action: "REPORT_SUBMITTED"
    ├─ Store submission_id
    ├─ Store record_count
    └─ Store timestamp
    ↓
Calculate Compliance Score
    ├─ Count on-time submissions
    ├─ Check alert acknowledgments
    ├─ Verify data quality
    └─ Update score
    ↓
Return Success Response
    ├─ submission_id
    ├─ record_count
    └─ message
    ↓
Frontend: Show Confirmation
    ├─ Display success message
    ├─ Show submission ID
    └─ Redirect to dashboard
```

### Excel Upload Flow

```
User Selects File
    ↓
Frontend: File Upload (pharmacy-reports.js)
    ├─ Read file with FileReader
    ├─ Parse with XLSX library
    └─ Extract data
    ↓
API Call: POST /api/pharmacy/reports/validate-excel
    ├─ file: <Excel file>
    └─ report_type: "anonymous|identified|aggregated"
    ↓
Backend: Validate Schema (pharmacy_report_routes.py)
    ├─ Read Excel file with openpyxl
    ├─ Extract column headers
    ├─ Get required columns from schema
    ├─ Check for missing columns
    ├─ Check for extra columns
    └─ Generate preview (first 10 rows)
    ↓
Validation Result
    ├─ If Valid:
    │  ├─ Return preview data
    │  ├─ Return total row count
    │  └─ Return column mapping
    │
    └─ If Invalid:
       ├─ Return error message
       ├─ List missing columns
       └─ Suggest template download
    ↓
Frontend: Display Result
    ├─ If Valid:
    │  ├─ Show success message
    │  ├─ Display preview table
    │  └─ Enable submit button
    │
    └─ If Invalid:
       ├─ Show error prompt
       ├─ List required columns
       └─ Disable submit button
    ↓
User Confirms & Submits
    ↓
API Call: POST /api/pharmacy/reports/submit
    ├─ report_type: "anonymous|identified|aggregated"
    ├─ entry_mode: "excel"
    └─ records: [Excel data]
    ↓
Backend: Process Records
    ├─ Validate each record
    ├─ Create model instances
    ├─ Commit to database
    ├─ Log submission
    └─ Update compliance score
    ↓
Return Success Response
    ↓
Frontend: Show Confirmation
```

---

## Schema Validation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCHEMA DEFINITIONS                        │
│                                                               │
│  SCHEMAS = {                                                 │
│    'anonymous': [                                            │
│      { key: 'drug_name', label: 'Drug Name',                │
│        type: 'text', required: true },                       │
│      { key: 'severity', label: 'Severity',                  │
│        type: 'select', options: [...], required: true },     │
│      ...                                                      │
│    ],                                                         │
│    'identified': [                                           │
│      ...anonymous fields...                                  │
│      { key: 'consent_verified', label: 'Consent Verified',  │
│        type: 'checkbox', required: true },                   │
│      ...                                                      │
│    ],                                                         │
│    'aggregated': [                                           │
│      { key: 'drug_name', label: 'Drug Name',                │
│        type: 'text', required: true },                       │
│      { key: 'total_dispensed', label: 'Total Dispensed',    │
│        type: 'number', required: true },                     │
│      ...                                                      │
│    ]                                                          │
│  }                                                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              FRONTEND VALIDATION (Client-side)               │
│                                                               │
│  1. Form Rendering                                           │
│     └─ renderFormFields() uses schema                        │
│                                                               │
│  2. Field Validation                                         │
│     ├─ Required field check                                  │
│     ├─ Data type validation                                  │
│     ├─ Format validation (dates, numbers)                    │
│     └─ Option validation (select fields)                     │
│                                                               │
│  3. Form Collection                                          │
│     └─ collectFormData() extracts values                     │
│                                                               │
│  4. Pre-submission Validation                                │
│     └─ validateFormData() checks all records                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              BACKEND VALIDATION (Server-side)                │
│                                                               │
│  1. Schema Lookup                                            │
│     └─ Get required columns from REPORT_SCHEMAS             │
│                                                               │
│  2. Record Validation                                        │
│     ├─ Check all required fields present                     │
│     ├─ Check no extra fields                                 │
│     ├─ Validate data types                                   │
│     └─ Validate field values                                 │
│                                                               │
│  3. Error Handling                                           │
│     ├─ Collect all errors                                    │
│     ├─ Return detailed error message                         │
│     └─ Suggest corrections                                   │
│                                                               │
│  4. Success Path                                             │
│     ├─ Create model instances                                │
│     ├─ Set field values                                      │
│     ├─ Commit to database                                    │
│     └─ Return submission ID                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## State Management

```
reportState = {
    reportType: 'anonymous',           // Current report type
    entryMode: 'manual',               // Current entry mode
    schema: [],                        // Current schema
    records: [],                       // Collected records
    excelData: null,                   // Excel data after validation
    columnMapping: null                // Column mapping for Excel
}

State Transitions:
    Initial State
        ↓
    User selects report type
        → reportState.reportType = 'anonymous|identified|aggregated'
        → renderFormFields()
        → updateStepIndicators()
        ↓
    User selects entry mode
        → reportState.entryMode = 'manual|excel'
        → updateEntryModeUI()
        → updateStepIndicators()
        ↓
    User enters data
        → collectFormData() or handleFileSelect()
        → reportState.records or reportState.excelData
        → updateSummary()
        ↓
    User submits
        → submitReport()
        → API call
        → Success/Error handling
        ↓
    User resets
        → resetForm()
        → Clear all state
        → Return to initial state
```

---

## Compliance Scoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  COMPLIANCE SCORING SYSTEM                   │
│                                                               │
│  Base Score: 100 points                                      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ POSITIVE BEHAVIORS (Increase Score)                 │    │
│  │                                                      │    │
│  │ On-time submission:        +5 points                │    │
│  │ Alert acknowledged:        +3 points                │    │
│  │ Severe ADR reported:       +10 points               │    │
│  │ Consistent reporting:      +2 points/month          │    │
│  │ Correct Excel schema:      +2 points                │    │
│  │ Complete fields:           +1 point                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ NEGATIVE BEHAVIORS (Decrease Score)                 │    │
│  │                                                      │    │
│  │ Late submission:           -5 points                │    │
│  │ Ignored alert:             -10 points               │    │
│  │ Invalid Excel upload:      -2 points                │    │
│  │ Missing required fields:   -3 points                │    │
│  │ Long reporting gap:        -5 points/month          │    │
│  │ Schema violation:          -2 points                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  Score Calculation:                                          │
│  Final Score = Base Score + Positive Actions - Negative      │
│  Range: 0-100 (capped)                                       │
│                                                               │
│  Status Mapping:                                             │
│  80-100: 🟢 Compliant                                        │
│  60-79:  🟡 Attention Required                               │
│  <60:    🔴 Non-compliant                                    │
│                                                               │
│  Update Frequency: Daily (automatic)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Error Handling Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ERROR HANDLING FLOW                        │
│                                                               │
│  Frontend Errors                                             │
│  ├─ Missing required fields                                  │
│  │  └─ Show inline error message                             │
│  ├─ Invalid data format                                      │
│  │  └─ Show field-level error                                │
│  ├─ File upload errors                                       │
│  │  └─ Show validation prompt                                │
│  └─ API errors                                               │
│     └─ Show alert with error message                         │
│                                                               │
│  Backend Errors                                              │
│  ├─ Invalid report type                                      │
│  │  └─ Return 400 with error message                         │
│  ├─ Schema validation failure                                │
│  │  └─ Return 400 with missing fields                        │
│  ├─ Database errors                                          │
│  │  └─ Return 500 with error message                         │
│  └─ Authentication errors                                    │
│     └─ Return 401 with error message                         │
│                                                               │
│  Error Response Format:                                      │
│  {                                                            │
│    "success": false,                                          │
│    "message": "Error description",                            │
│    "details": {...}  // Optional                             │
│  }                                                            │
│                                                               │
│  User Feedback                                               │
│  ├─ Non-blocking prompts (white cards)                       │
│  ├─ Clear error messages                                     │
│  ├─ Actionable suggestions                                   │
│  └─ Guidance for correction                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                           │
│                                                               │
│  1. Authentication                                           │
│     ├─ Session-based authentication                          │
│     ├─ User ID from session                                  │
│     └─ Verified on each request                              │
│                                                               │
│  2. Authorization                                            │
│     ├─ User role verification                                │
│     ├─ Pharmacy ownership check                              │
│     └─ Data access control                                   │
│                                                               │
│  3. Input Validation                                         │
│     ├─ Frontend validation (UX)                              │
│     ├─ Backend validation (Security)                         │
│     ├─ Schema enforcement                                    │
│     └─ Type checking                                         │
│                                                               │
│  4. Data Protection                                          │
│     ├─ Soft deletes (no hard deletes)                        │
│     ├─ Audit trail (all actions logged)                      │
│     ├─ Timestamps (created_at, updated_at)                   │
│     └─ User tracking (pharmacy_id)                           │
│                                                               │
│  5. API Security                                             │
│     ├─ CORS restrictions                                     │
│     ├─ Content-Type validation                               │
│     ├─ Rate limiting (optional)                              │
│     └─ HTTPS (production)                                    │
│                                                               │
│  6. Database Security                                        │
│     ├─ Parameterized queries (SQLAlchemy)                    │
│     ├─ SQL injection prevention                              │
│     ├─ Foreign key constraints                               │
│     └─ Indexed queries                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Optimization

```
┌─────────────────────────────────────────────────────────────┐
│                  PERFORMANCE STRATEGIES                      │
│                                                               │
│  Frontend Optimization                                       │
│  ├─ Lazy form rendering                                      │
│  ├─ Client-side validation first                             │
│  ├─ Async file processing                                    │
│  ├─ Efficient DOM updates                                    │
│  └─ Minimal re-renders                                       │
│                                                               │
│  Backend Optimization                                        │
│  ├─ Efficient database queries                               │
│  ├─ Indexed fields (pharmacy_id, created_at)                 │
│  ├─ Pagination support                                       │
│  ├─ Batch inserts                                            │
│  └─ Connection pooling                                       │
│                                                               │
│  Data Optimization                                           │
│  ├─ Polymorphic inheritance (single table)                   │
│  ├─ JSON fields for flexible data                            │
│  ├─ Soft deletes (no cleanup)                                │
│  └─ Audit trail (separate table)                             │
│                                                               │
│  Caching Strategy                                            │
│  ├─ Compliance score (daily update)                          │
│  ├─ Schema definitions (static)                              │
│  ├─ User session (in-memory)                                 │
│  └─ Template download (on-demand)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT STRUCTURE                       │
│                                                               │
│  Frontend Files                                              │
│  ├─ templates/pharmacy/reports.html                          │
│  ├─ static/js/pharmacy-reports.js                            │
│  ├─ static/css/style.css                                     │
│  └─ External: XLSX library (CDN)                             │
│                                                               │
│  Backend Files                                               │
│  ├─ pv_backend/routes/pharmacy_report_routes.py              │
│  ├─ pv_backend/models/pharmacy_report.py                     │
│  ├─ pv_backend/services/audit_service.py                     │
│  └─ pv_backend/app.py (blueprint registration)               │
│                                                               │
│  Database                                                    │
│  ├─ pharmacy_reports (base table)                            │
│  ├─ anonymous_reports (inherits)                             │
│  ├─ identified_reports (inherits)                            │
│  ├─ aggregated_reports (inherits)                            │
│  └─ submission_logs (audit trail)                            │
│                                                               │
│  Configuration                                               │
│  ├─ .env (environment variables)                             │
│  ├─ pv_backend/config.py (app config)                        │
│  └─ CORS settings (app.py)                                   │
│                                                               │
│  Documentation                                               │
│  ├─ docs/PHARMACY_REPORTS_MODULE.md                          │
│  ├─ docs/PHARMACY_REPORTS_QUICK_START.md                     │
│  ├─ docs/COMPLIANCE_SCORING_GUIDE.md                         │
│  └─ docs/PHARMACY_REPORTS_ARCHITECTURE.md                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                   SYSTEM INTEGRATIONS                        │
│                                                               │
│  User Authentication                                         │
│  └─ session['user_id'] from login system                     │
│                                                               │
│  Database                                                    │
│  └─ SQLAlchemy ORM with Flask-SQLAlchemy                     │
│                                                               │
│  Audit Service                                               │
│  └─ log_action() for submission tracking                     │
│                                                               │
│  Pharmacy Dashboard                                          │
│  └─ Redirect after submission                                │
│                                                               │
│  External Libraries                                          │
│  ├─ XLSX (Excel handling)                                    │
│  ├─ Lucide (Icons)                                           │
│  ├─ Flask (Web framework)                                    │
│  └─ SQLAlchemy (ORM)                                         │
│                                                               │
│  Future Integrations                                         │
│  ├─ Firebase (optional)                                      │
│  ├─ Email service (notifications)                            │
│  ├─ Regulatory systems (reporting)                           │
│  └─ Analytics platform (insights)                            │
└─────────────────────────────────────────────────────────────┘
```

---

**Last Updated**: January 28, 2024
**Version**: 1.0
