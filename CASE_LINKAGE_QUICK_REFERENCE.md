# Case Linkage System - Quick Reference Guide

## What Was Implemented?

A **Case Matching Engine** that automatically detects duplicate/similar adverse event cases when doctors submit new reports.

---

## Key Components

### 1. **Case Matching Service** 📦
**Location:** `pv_backend/services/case_matching.py`

Core matching algorithm using:
- **Fuzzy String Matching** (Python SequenceMatcher)
- **Weighted Similarity Scoring**
- **Configurable Thresholds**

**Usage:**
```python
from pv_backend.services.case_matching import match_new_case

new_case = {
    'drug_name': 'Aspirin',
    'symptoms': 'headache, fever',
    'age': 45,
    'gender': 'Female'
}

results = match_new_case(new_case, existing_patients, threshold=0.75)
```

---

### 2. **New Database Fields** 💾
**Modified:** `models.py` Patient Model

```python
linked_case_id      # Links to parent case if duplicate
match_score         # Similarity score (0-1)
case_status         # Active, Linked, or Discarded
match_notes         # Reason for decision
```

---

### 3. **API Endpoints** 🔗

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/cases/match` | POST | Check for matching cases |
| `/api/cases/link` | POST | Link case as duplicate |
| `/api/cases/discard` | POST | Mark case as discarded |
| `/api/cases/<id>/details` | GET | Get case details |

---

### 4. **Doctor Report Page UI** 🎨
**Modified:** `templates/doctor/report.html`

**New Features:**
- ✅ Automatic duplicate detection
- ✅ Interactive modal with matching cases
- ✅ Three action buttons (Accept/Link/Discard)
- ✅ Detailed similarity score breakdown
- ✅ Color-coded confidence levels

---

## How It Works

```
Doctor Submits Report
        ↓
System Checks for Matches
        ↓
        ├─ NO MATCHES → Submit as new case ✓
        │
        └─ MATCHES FOUND → Show Modal
              ↓
              ├─ "Add as New" → Creates new case
              ├─ "Link" → Marks as duplicate (linked)
              └─ "Discard" → Marks as duplicate (discarded)
```

---

## Similarity Score Breakdown

**How cases are scored (weighted):**

1. **Drug Name** (40% weight)
   - Must match exactly
   - Case-insensitive
   - Example: "Aspirin" = "aspirin" ✓

2. **Symptoms** (40% weight)
   - Fuzzy text matching
   - Handles typos/variations
   - Example: "headache" ≈ "head ache" (high match)

3. **Demographics** (15% weight)
   - Age proximity (±10 years window)
   - Gender match
   - Example: Age 45 matches 50 with 0.75 score

4. **Time Recency** (5% weight)
   - Recent cases weighted higher
   - 0-30 day window most relevant
   - Example: Report from today > report from 6 months ago

---

## Thresholds Explained

| Score Range | Confidence | Recommendation |
|------------|-----------|-----------------|
| ≥ 90% | Very High | **DISCARD** - Auto-recommend discard |
| 80-89% | High | **REVIEW** - Likely duplicate |
| 75-79% | Medium | **REVIEW** - Possible duplicate |
| < 75% | Low | **ACCEPT** - Treat as new case |

---

## Example: Case Matching Scenario

### Case 1 (Existing in Database)
- **Drug:** Aspirin
- **Symptoms:** "severe headache, dizziness"
- **Age:** 45
- **Gender:** Female
- **Reported:** 2026-01-25

### Case 2 (New Submission)
- **Drug:** Aspirin
- **Symptoms:** "headache, dizzy"
- **Age:** 47
- **Gender:** Female

### Matching Result:
```
Drug Match:        100% (exact match)
Symptom Match:     78%  (similar)
Demographics:      75%  (age 47 vs 45 + gender match)
Time Recency:      85%  (recent)
─────────────────────────
OVERALL SCORE:     82% ✓
Recommendation:    REVIEW - Likely duplicate
```

---

## API Examples

### Check for Matches
```bash
curl -X POST http://localhost:5000/api/cases/match \
  -H "Content-Type: application/json" \
  -d '{
    "drugName": "Aspirin",
    "symptoms": "headache, fever",
    "age": 45,
    "gender": "Female",
    "threshold": 0.75
  }'
```

### Link to Existing Case
```bash
curl -X POST http://localhost:5000/api/cases/link \
  -H "Content-Type: application/json" \
  -d '{
    "newCaseId": "DR-1234567890",
    "linkedCaseId": "PT-1001",
    "matchScore": 0.82,
    "notes": "Duplicate confirmed by doctor"
  }'
```

### Discard Duplicate
```bash
curl -X POST http://localhost:5000/api/cases/discard \
  -H "Content-Type: application/json" \
  -d '{
    "caseId": "DR-1234567890",
    "reason": "Exact duplicate of PT-1001"
  }'
```

---

## Configuration

### Change Default Threshold
In `case_matching.py`:
```python
engine = CaseMatchingEngine(threshold=0.80)  # Default 0.75
```

### Adjust Age Window
In `case_matching.py`:
```python
age_similarity = self.calculate_age_similarity(
    new_case.get('age'),
    existing_case.age,
    max_age_diff=15  # Change from default 10
)
```

---

## Benefits

✅ **Reduces Duplicates** - Prevents same case reported multiple times  
✅ **Improves Data Quality** - Cleaner adverse event database  
✅ **Fast Processing** - No external API calls needed  
✅ **User Control** - Doctors review before decisions  
✅ **Audit Trail** - All linkages documented  
✅ **Configurable** - Thresholds adjustable as needed  

---

## Files Modified/Created

| File | Type | Status |
|------|------|--------|
| `pv_backend/services/case_matching.py` | NEW | ✅ Created |
| `models.py` | MODIFIED | ✅ Updated |
| `app.py` | MODIFIED | ✅ Updated (imports + 4 endpoints) |
| `templates/doctor/report.html` | MODIFIED | ✅ Updated (UI + JS) |

---

## Testing Checklist

- [ ] Install dependencies (if needed)
- [ ] Run migrations (`python app.py` - creates tables)
- [ ] Navigate to `/doctor/report`
- [ ] Fill out form with: Aspirin, headache, age 45
- [ ] Submit and check for matches
- [ ] Submit similar case - should show modal
- [ ] Test Link/Discard actions
- [ ] Verify database updates

---

## Error Handling

### What if matching fails?
- System falls back to standard submission
- Case submitted as new anyway
- Error logged for debugging

### What if database is empty?
- No matches found → "ACCEPT" recommendation
- Case submitted as first case for that drug

### What if age/gender missing?
- System handles gracefully
- Uses available data for matching
- Doesn't penalize missing fields

---

## No External Dependencies

✅ No LLM API required  
✅ No paid services  
✅ No network requests to third parties  
✅ Fast local processing  
✅ Works offline  

---

## Next Steps

1. **Test in Doctor Report Page:**
   - Go to `/doctor/report`
   - Fill form and submit
   - Review match modal

2. **Monitor Duplicate Rate:**
   - Check database for linked cases
   - Adjust threshold if needed

3. **Gather User Feedback:**
   - Doctors review recommendations
   - Refine threshold based on feedback

4. **Future Enhancements:**
   - Add ML-based matching
   - Patient de-duplication
   - Bulk case review

---

**Questions?** Check `CASE_LINKAGE_IMPLEMENTATION.md` for detailed documentation.
