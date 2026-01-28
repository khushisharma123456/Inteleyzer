# Pharmacy Reports - Quick Start Guide

## 🚀 Getting Started

### Step 1: Navigate to Reports
Go to **Pharmacy Dashboard** → **Reports**

### Step 2: Select Report Type

Choose one of three options:

#### 🔒 Anonymous Data (DEFAULT)
- **Best for**: General adverse event reporting
- **No identifiers**: Safe for all pharmacies
- **Fields**: Drug name, dosage, reaction, severity, age group
- **Use when**: You don't have patient consent

#### 👤 Data with Identity (RESTRICTED)
- **Best for**: Follow-up investigations, case linkage
- **Limited identifiers**: Hospital/doctor reference only
- **⚠️ Warning**: Only submit if consent is available
- **Use when**: You need to coordinate with hospitals

#### 📊 Aggregated / Disease Analysis
- **Best for**: Trend analysis, pattern detection
- **Summary only**: No individual records
- **Excel-only**: Cannot use manual entry
- **Use when**: Reporting monthly/quarterly summaries

### Step 3: Choose Entry Mode

#### 📝 Manual Entry
- Fill in form fields
- Add multiple records one by one
- Best for: Small submissions (1-10 records)

#### 📁 Excel Upload
- Download template
- Fill in Excel file
- Upload and submit
- Best for: Bulk submissions (10+ records)

---

## 📝 Manual Entry Workflow

1. **Select Report Type** → **Select Manual Entry**
2. **Fill in Record 1**:
   - Drug Name (required)
   - Batch Number (optional)
   - Dosage Form (required)
   - Date of Dispensing (required)
   - Reaction Category (required)
   - Severity (required)
   - Reaction Outcome (optional)
   - Age Group (required)
   - Gender (optional)
   - Additional Notes (optional)

3. **Add More Records** (optional):
   - Click "Add Another Record"
   - Fill in next record
   - Repeat as needed

4. **Review Summary**:
   - Check report type
   - Check entry mode
   - Check total records
   - Check data scope

5. **Confirm & Submit**:
   - ☑️ Check "I confirm the data is accurate and compliant"
   - Click "Submit Report"

---

## 📁 Excel Upload Workflow

### 1. Download Template
- Click "Download Excel Template"
- File name: `pharmacy_report_template_[type].xlsx`

### 2. Fill in Excel
- **Row 1**: Column headers (already filled)
- **Row 2**: Description (already filled)
- **Row 3+**: Your data

**Important**:
- ✅ Keep column names exactly as shown
- ✅ Fill in all required columns
- ❌ Don't add extra columns
- ❌ Don't remove columns
- ❌ Don't change column order

### 3. Upload File
- Click upload area or drag-and-drop
- Wait for validation

### 4. Check Validation Result

**✓ If Valid**:
- Green success message
- Preview of first 10 rows
- Click "Submit Report"

**✗ If Invalid**:
- Red error message
- List of missing columns
- Download template again
- Fix your file
- Re-upload

### 5. Review & Submit
- Check preview table
- Confirm data looks correct
- Click "Submit Report"

---

## ✅ Required Fields by Report Type

### Anonymous Data
- Drug Name
- Dosage Form
- Date of Dispensing
- Reaction Category
- Severity
- Age Group

### Data with Identity
- Drug Name
- Dosage Form
- Date of Dispensing
- Reaction Category
- Severity
- Age Group
- **Consent Verified** (checkbox)

### Aggregated / Disease Analysis
- Drug Name
- Total Dispensed
- Total Reactions Reported
- Mild Count
- Moderate Count
- Severe Count
- Reporting Period Start
- Reporting Period End

---

## 🎯 Best Practices

### Data Quality
- ✅ Use consistent drug names
- ✅ Fill in all required fields
- ✅ Use correct date format (YYYY-MM-DD)
- ✅ Be specific in reaction descriptions

### Compliance
- ✅ Submit on time
- ✅ Acknowledge safety alerts
- ✅ Use correct report type
- ✅ Verify consent before submitting identified data

### Excel Tips
- ✅ Download fresh template each time
- ✅ Don't modify column headers
- ✅ Save as .xlsx (not .xls)
- ✅ Test with small file first

---

## ⚠️ Common Issues

### "Invalid Excel Format"
**Problem**: Column names don't match template
**Solution**: 
1. Download fresh template
2. Copy your data into new template
3. Re-upload

### "Missing Required Fields"
**Problem**: You didn't fill in all required columns
**Solution**:
1. Check the error message
2. Add missing columns to Excel
3. Fill in data for all rows
4. Re-upload

### "Validation Failed"
**Problem**: Data format is incorrect
**Solution**:
1. Check date format (YYYY-MM-DD)
2. Check severity values (mild, moderate, severe)
3. Check age group values (pediatric, adult, elderly, etc.)
4. Fix and re-upload

---

## 📊 After Submission

### Confirmation
- You'll see: "Success! Submitted X records"
- Submission ID: `SUB-20240128193415`
- Save this ID for your records

### Submission History
- Go to **Reports** → **Submission History**
- See all your past submissions
- Check status: Submitted / Under Review / Flagged

### Compliance Score
- View your compliance status
- 🟢 Compliant (80-100)
- 🟡 Attention Required (60-79)
- 🔴 Non-compliant (<60)

---

## 🆘 Need Help?

### Contact Support
- Email: support@inteleyzer.com
- Phone: +1-XXX-XXX-XXXX
- Chat: Available in dashboard

### Documentation
- Full guide: See PHARMACY_REPORTS_MODULE.md
- API docs: See API_DOCUMENTATION.md
- FAQ: See PHARMACY_REPORTS_FAQ.md

---

## 📋 Submission Checklist

Before clicking Submit:

- [ ] Report type selected correctly
- [ ] All required fields filled
- [ ] Data is accurate
- [ ] No patient contact information (if anonymous)
- [ ] Consent verified (if identified)
- [ ] Dates in correct format
- [ ] Confirmation checkbox checked
- [ ] Ready to submit

---

## 🔐 Data Privacy

### Anonymous Data
- No identifiers stored
- Safe for all pharmacies
- Used for analytics

### Identified Data
- Limited identifiers only
- Requires consent
- Used for case linkage
- Secure storage

### Aggregated Data
- Summary counts only
- No individual records
- Used for trends
- Regulatory reporting

---

**Last Updated**: January 28, 2024
**Version**: 1.0
