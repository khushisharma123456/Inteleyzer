# Case Linkage & Duplicate Detection System - Implementation Summary

## Overview
Implemented a comprehensive **Case Matching Engine** that automatically detects duplicate/similar adverse event cases before adding new cases to the database using **fuzzy string matching and weighted similarity scoring**.

---

## Files Created & Modified

### 1. **Case Matching Service** ✅
**File:** `pv_backend/services/case_matching.py` (New)

**Features:**
- **CaseMatchingEngine Class** - Core matching algorithm
  - `calculate_text_similarity()` - Fuzzy string matching for symptoms using SequenceMatcher
  - `calculate_age_similarity()` - Age-based proximity scoring (configurable window, default 10 years)
  - `calculate_case_similarity()` - Comprehensive scoring with 4 weighted components
  - `find_matching_cases()` - Finds all matching cases above threshold
  
- **Weighted Scoring Breakdown:**
  - Drug Name Match: **40%** (exact match = high priority)
  - Symptom Similarity: **40%** (fuzzy text matching)
  - Demographics (Age + Gender): **15%** (proximity matching)
  - Time Recency: **5%** (recent reports weighted higher)
  - **Default Threshold:** 0.75 (75% match needed)

- **Utility Functions:**
  - `match_new_case()` - Convenience function for matching
  - `should_accept_case()` - Auto-decision logic (ACCEPT/REVIEW/DISCARD)

---

### 2. **Database Model Updates** ✅
**File:** `models.py` (Modified)

**New Fields Added to Patient Model:**
```python
linked_case_id = db.Column(db.String(20), db.ForeignKey('patient.id'), nullable=True)
match_score = db.Column(db.Float, nullable=True)  # 0-1
case_status = db.Column(db.String(20), default='Active')  # Active, Linked, Discarded
match_notes = db.Column(db.Text, nullable=True)  # Reason for linkage/decision
```

**Status Values:**
- `Active` - Standalone case in system
- `Linked` - Linked to another case as potential duplicate
- `Discarded` - Marked as duplicate, not counted in statistics

---

### 3. **API Endpoints** ✅
**File:** `app.py` (Modified)

#### Endpoint 1: Check for Matching Cases
```
POST /api/cases/match
```
**Request:**
```json
{
  "drugName": "Aspirin",
  "symptoms": "headache, fever",
  "age": 45,
  "gender": "Female",
  "threshold": 0.75  // Optional
}
```

**Response:**
```json
{
  "success": true,
  "matches": [
    {
      "case_id": "PT-1001",
      "patient_name": "John Doe",
      "drug_name": "Aspirin",
      "symptoms": "headache",
      "age": 47,
      "gender": "Male",
      "created_at": "2026-01-25T10:30:00",
      "similarity_score": 0.82,
      "breakdown": {
        "drug": 1.0,
        "symptoms": 0.75,
        "demographics": 0.6,
        "time_recency": 0.9
      },
      "is_match": true,
      "confidence": "High"
    }
  ],
  "total_matches": 1,
  "has_exact_match": false,
  "recommendation": "REVIEW - Likely duplicate, review before accepting",
  "action": "REVIEW",
  "reason": "High similarity (82%) with case PT-1001"
}
```

---

#### Endpoint 2: Link Cases
```
POST /api/cases/link
```
**Request:**
```json
{
  "newCaseId": "DR-1234567890",
  "linkedCaseId": "PT-1001",
  "matchScore": 0.82,
  "notes": "Doctor reviewed and confirmed duplicate"
}
```

**Result:** New case marked as "Linked" with match details stored

---

#### Endpoint 3: Discard Case
```
POST /api/cases/discard
```
**Request:**
```json
{
  "caseId": "DR-1234567890",
  "reason": "Identified as duplicate during review"
}
```

**Result:** Case marked as "Discarded"

---

#### Endpoint 4: Get Case Details
```
GET /api/cases/{case_id}/details
```

**Returns:** Full case info including linkage and match score

---

### 4. **Doctor Report UI Enhancement** ✅
**File:** `templates/doctor/report.html` (Modified)

#### Features Added:

**A. Case Matching Modal Dialog**
- Displays when matching cases are found
- Shows similarity score with color-coded confidence levels
- Detailed score breakdown (drug%, symptom%, demographics%, time%)
- Interactive case selection

**B. Three Action Buttons:**
1. **Add as New Case** - Ignore matches, submit as new
2. **Link to Selected Case** - Link as potential duplicate (requires case selection)
3. **Discard Duplicate** - Mark case as discarded

**C. Recommendation System:**
- `ACCEPT` - Green: No matches found
- `REVIEW` - Orange: Likely duplicate, manual review recommended (80%+ match)
- `DISCARD` - Red: Very likely duplicate, auto-discard suggestion (90%+ match)

**D. Smart Form Validation:**
- Form captures: Drug Name, Symptoms, Age, Gender, Demographics
- Automatically checks for matches before submission
- Shows detailed matching analysis before user decides

---

## How It Works - User Flow

### Step 1: Doctor Submits a New Case
Doctor fills out the report form with:
- Patient name, age, gender
- Drug name
- Observed symptoms
- Other clinical details

### Step 2: Automatic Matching Check
When doctor clicks "Submit", the system:
1. Extracts key fields (drug, symptoms, age, gender)
2. Queries database for cases with same drug
3. Calculates similarity score for each match
4. Returns results with confidence levels

### Step 3: Modal Presentation
If matches found:
- Modal displays all similar cases
- Shows detailed score breakdown
- Provides recommendation (ACCEPT/REVIEW/DISCARD)
- Doctor can select one to link or discard

### Step 4: Doctor Decision
Doctor has three options:
- **Add as New** - Creates new independent case
- **Link to Existing** - Marks as potential duplicate of selected case
- **Discard** - Removes as confirmed duplicate

### Step 5: Database Update
- New case created with status: "Active", "Linked", or "Discarded"
- Match score and linked case ID stored
- Case linkage preserved for reference

---

## Matching Algorithm Details

### Text Similarity (Fuzzy Matching)
- Uses Python's `SequenceMatcher` (not LLM-based)
- Handles typos, abbreviations, variations
- Example: "headache" vs "head ache" = high match
- Case-insensitive, whitespace-normalized

### Age Similarity
- Default window: ±10 years
- Score degrades smoothly outside window
- Example: 45 years matches 50 years = 0.75 score

### Time Recency
- Recent cases (0-30 days) weighted higher
- Older cases less likely to be duplicates
- Useful for detecting immediate duplicate reports

### Exact Drug Matching
- Drug name must match exactly (after normalization)
- Case-insensitive comparison
- Essential for pharmacovigilance accuracy

---

## Configuration Options

### Threshold (Default: 0.75)
```python
# Can be customized per request
engine = CaseMatchingEngine(threshold=0.80)
```

### Auto-Discard Threshold (Default: 0.90)
Cases scoring ≥90% automatically recommended for discard

### Age Window (Default: 10 years)
```python
age_similarity = calculate_age_similarity(age1, age2, max_age_diff=15)
```

---

## Benefits

✅ **Prevents Duplicate Cases** - Reduces noise in safety data  
✅ **Improves Data Quality** - Links related adverse events  
✅ **Pharmacovigilance** - Better pattern detection  
✅ **User Control** - Doctors review before action  
✅ **Audit Trail** - All linkages documented  
✅ **No External Dependencies** - Fast, local processing  
✅ **Configurable** - Thresholds adjustable per needs  

---

## Technical Stack

- **Backend:** Flask + SQLAlchemy
- **Matching:** Python SequenceMatcher (fuzzy matching)
- **Frontend:** Vanilla JavaScript + HTML/CSS
- **Database:** SQLite (expandable to PostgreSQL)
- **No External APIs** - Completely self-contained

---

## Future Enhancements

1. **ML-Based Matching** - Train model on labeled case pairs
2. **Natural Language Processing** - Semantic symptom matching
3. **Patient De-duplication** - Link same patients across cases
4. **Automated Merging** - Auto-merge high-confidence duplicates
5. **Analytics Dashboard** - Duplicate rate metrics
6. **Bulk Case Review** - Batch processing interface

---

## Testing

To test the implementation:

1. **Start the Flask app:**
   ```bash
   python app.py
   ```

2. **Navigate to doctor report page:**
   ```
   http://localhost:5000/doctor/report
   ```

3. **Submit a case with:**
   - Drug: "Aspirin"
   - Symptoms: "headache, dizziness"
   - Age: 45
   - Gender: Female

4. **Submit another similar case** to trigger matching modal

5. **Review suggestions** and select action (Link/Discard/Accept)

---

## Summary

✅ **Implemented:** Full case linkage system with threshold-based deduplication  
✅ **No LLM Required:** Uses efficient fuzzy matching algorithms  
✅ **User-Friendly:** Interactive modal with clear recommendations  
✅ **Audit Trail:** All decisions logged and reversible  
✅ **Extensible:** Easy to adjust thresholds or add new similarity metrics  
