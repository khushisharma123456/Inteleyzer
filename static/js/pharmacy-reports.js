/**
 * Pharmacy Reports Module
 * Handles form generation, validation, Excel upload, and submission
 */

// Global state
let reportState = {
    reportType: 'anonymous',
    entryMode: 'manual',
    records: [],
    schema: null,
    excelData: null,
    columnMapping: null
};

// Schema definitions (mirrors backend)
const SCHEMAS = {
    anonymous: [
        { key: 'drug_name', label: 'Drug Name', type: 'text', required: true },
        { key: 'batch_lot_number', label: 'Batch / Lot Number', type: 'text', required: false },
        { key: 'dosage_form', label: 'Dosage Form', type: 'text', required: true },
        { key: 'date_of_dispensing', label: 'Date of Dispensing', type: 'date', required: true },
        { key: 'reaction_category', label: 'Reaction Category', type: 'text', required: true },
        { key: 'severity', label: 'Severity', type: 'select', options: ['mild', 'moderate', 'severe'], required: true },
        { key: 'reaction_outcome', label: 'Reaction Outcome', type: 'select', options: ['recovered', 'recovering', 'not_recovered', 'fatal', 'unknown'], required: false },
        { key: 'age_group', label: 'Age Group', type: 'select', options: ['pediatric', 'adolescent', 'adult', 'elderly', 'unknown'], required: true },
        { key: 'gender', label: 'Gender', type: 'select', options: ['male', 'female', 'other', 'not_specified'], required: false },
        { key: 'additional_notes', label: 'Additional Notes', type: 'textarea', required: false }
    ],
    identified: [
        { key: 'drug_name', label: 'Drug Name', type: 'text', required: true },
        { key: 'batch_lot_number', label: 'Batch / Lot Number', type: 'text', required: false },
        { key: 'dosage_form', label: 'Dosage Form', type: 'text', required: true },
        { key: 'date_of_dispensing', label: 'Date of Dispensing', type: 'date', required: true },
        { key: 'reaction_category', label: 'Reaction Category', type: 'text', required: true },
        { key: 'severity', label: 'Severity', type: 'select', options: ['mild', 'moderate', 'severe'], required: true },
        { key: 'reaction_outcome', label: 'Reaction Outcome', type: 'select', options: ['recovered', 'recovering', 'not_recovered', 'fatal', 'unknown'], required: false },
        { key: 'age_group', label: 'Age Group', type: 'select', options: ['pediatric', 'adolescent', 'adult', 'elderly', 'unknown'], required: true },
        { key: 'gender', label: 'Gender', type: 'select', options: ['male', 'female', 'other', 'not_specified'], required: false },
        { key: 'internal_case_id', label: 'Internal Case ID', type: 'text', required: false },
        { key: 'treating_hospital_reference', label: 'Treating Hospital / Doctor Reference', type: 'text', required: false },
        { key: 'treating_doctor_name', label: 'Treating Doctor Name', type: 'text', required: false },
        { key: 'consent_verified', label: 'Consent Verified', type: 'checkbox', required: true },
        { key: 'consent_date', label: 'Consent Date', type: 'date', required: false },
        { key: 'additional_notes', label: 'Additional Notes', type: 'textarea', required: false }
    ],
    aggregated: [
        { key: 'drug_name', label: 'Drug Name', type: 'text', required: true },
        { key: 'total_dispensed', label: 'Total Dispensed', type: 'number', required: true },
        { key: 'total_reactions_reported', label: 'Total Reactions Reported', type: 'number', required: true },
        { key: 'mild_count', label: 'Mild Count', type: 'number', required: true },
        { key: 'moderate_count', label: 'Moderate Count', type: 'number', required: true },
        { key: 'severe_count', label: 'Severe Count', type: 'number', required: true },
        { key: 'reporting_period_start', label: 'Reporting Period Start', type: 'date', required: true },
        { key: 'reporting_period_end', label: 'Reporting Period End', type: 'date', required: true },
        { key: 'analysis_notes', label: 'Analysis Notes', type: 'textarea', required: false }
    ]
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    renderFormFields();
    updateStepIndicators();
});

function initializeEventListeners() {
    // Report type selector
    document.querySelectorAll('input[name="report_type"]').forEach(radio => {
        radio.addEventListener('change', function() {
            reportState.reportType = this.value;
            updateReportTypeUI();
            renderFormFields();
            updateStepIndicators();
        });
    });
    
    // Entry mode selector
    document.querySelectorAll('input[name="entry_mode"]').forEach(radio => {
        radio.addEventListener('change', function() {
            reportState.entryMode = this.value;
            updateEntryModeUI();
            updateStepIndicators();
        });
    });
    
    // File upload drag and drop
    const fileUploadArea = document.getElementById('fileUploadArea');
    if (fileUploadArea) {
        fileUploadArea.addEventListener('click', () => document.getElementById('excelFile').click());
        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('dragover');
        });
        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('dragover');
        });
        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('excelFile').files = files;
                handleFileSelect({ target: { files } });
            }
        });
    }
    
    // Confirmation checkbox
    document.getElementById('confirmCheckbox')?.addEventListener('change', function() {
        document.getElementById('submitBtn').disabled = !this.checked;
    });
}

function updateReportTypeUI() {
    // Update selector cards
    document.querySelectorAll('.selector-card').forEach(card => {
        const radio = card.querySelector('input[type="radio"]');
        if (radio.value === reportState.reportType) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
    
    // Show/hide warning for identified data
    const warning = document.getElementById('identifiedWarning');
    if (reportState.reportType === 'identified') {
        warning.classList.add('show');
    } else {
        warning.classList.remove('show');
    }
    
    // Disable Excel mode for non-aggregated types (or allow for all)
    // For now, allow all types to use both modes
}

function updateEntryModeUI() {
    // Update mode cards
    document.querySelectorAll('.mode-card').forEach(card => {
        const radio = card.querySelector('input[type="radio"]');
        if (radio.value === reportState.entryMode) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
    
    // Show/hide sections
    const manualSection = document.getElementById('manualEntrySection');
    const excelSection = document.getElementById('excelUploadSection');
    
    if (reportState.entryMode === 'manual') {
        manualSection.style.display = 'block';
        excelSection.style.display = 'none';
        document.getElementById('summaryBox').style.display = 'block';
    } else {
        manualSection.style.display = 'none';
        excelSection.style.display = 'block';
        document.getElementById('summaryBox').style.display = 'none';
    }
}

function updateStepIndicators() {
    // Update step badges based on state
    const badges = document.querySelectorAll('.step-badge');
    badges[0].classList.add('active');
    badges[0].classList.remove('completed');
    
    if (reportState.reportType) {
        badges[0].classList.remove('active');
        badges[0].classList.add('completed');
        badges[1].classList.add('active');
    }
    
    if (reportState.entryMode) {
        badges[1].classList.remove('active');
        badges[1].classList.add('completed');
        badges[2].classList.add('active');
    }
}

// ============================================================================
// FORM RENDERING
// ============================================================================

function renderFormFields() {
    const container = document.getElementById('formFieldsContainer');
    if (!container) return;
    
    container.innerHTML = '';
    
    const schema = SCHEMAS[reportState.reportType];
    reportState.schema = schema;
    
    // Create first form row
    addFormRow();
}

function addFormRow() {
    const container = document.getElementById('formFieldsContainer');
    if (!container) return;
    
    const rowIndex = container.querySelectorAll('.form-row').length;
    const schema = SCHEMAS[reportState.reportType];
    
    const rowDiv = document.createElement('div');
    rowDiv.className = 'form-row';
    rowDiv.dataset.rowIndex = rowIndex;
    
    let html = `
        <div class="form-row-header">
            <div class="form-row-title">Record ${rowIndex + 1}</div>
            ${rowIndex > 0 ? `<button type="button" class="form-row-remove" onclick="removeFormRow(${rowIndex})">Remove</button>` : ''}
        </div>
    `;
    
    schema.forEach(field => {
        html += renderFormField(field, rowIndex);
    });
    
    rowDiv.innerHTML = html;
    container.appendChild(rowDiv);
}

function removeFormRow(rowIndex) {
    const container = document.getElementById('formFieldsContainer');
    const rows = container.querySelectorAll('.form-row');
    if (rows[rowIndex]) {
        rows[rowIndex].remove();
    }
}

function renderFormField(field, rowIndex) {
    const fieldId = `${field.key}_${rowIndex}`;
    const required = field.required ? '<span class="required">*</span>' : '';
    
    let input = '';
    
    switch (field.type) {
        case 'text':
        case 'number':
            input = `<input type="${field.type}" id="${fieldId}" name="${field.key}" placeholder="Enter ${field.label.toLowerCase()}">`;
            break;
        
        case 'date':
            input = `<input type="date" id="${fieldId}" name="${field.key}">`;
            break;
        
        case 'select':
            input = `<select id="${fieldId}" name="${field.key}">
                <option value="">-- Select ${field.label} --</option>
                ${field.options.map(opt => `<option value="${opt}">${opt.charAt(0).toUpperCase() + opt.slice(1)}</option>`).join('')}
            </select>`;
            break;
        
        case 'checkbox':
            input = `<div style="display: flex; align-items: center; gap: 0.5rem;">
                <input type="checkbox" id="${fieldId}" name="${field.key}">
                <label for="${fieldId}" style="margin: 0;">Consent has been verified</label>
            </div>`;
            break;
        
        case 'textarea':
            input = `<textarea id="${fieldId}" name="${field.key}" placeholder="Enter ${field.label.toLowerCase()}"></textarea>`;
            break;
    }
    
    return `
        <div class="form-group">
            <label for="${fieldId}">${field.label} ${required}</label>
            ${input}
        </div>
    `;
}

// ============================================================================
// EXCEL HANDLING
// ============================================================================

function downloadTemplate() {
    const reportType = reportState.reportType;
    const schema = SCHEMAS[reportType];
    
    // Create workbook
    const wb = XLSX.utils.book_new();
    
    // Create header row
    const headers = schema.map(f => f.label);
    const descRow = schema.map(f => f.required ? 'Required' : 'Optional');
    
    const ws = XLSX.utils.aoa_to_sheet([headers, descRow]);
    
    // Set column widths
    ws['!cols'] = schema.map(() => ({ wch: 20 }));
    
    XLSX.utils.book_append_sheet(wb, ws, 'Data');
    XLSX.writeFile(wb, `pharmacy_report_template_${reportType}.xlsx`);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('report_type', reportState.reportType);
    
    // Show loading state
    const prompt = document.getElementById('validationPrompt');
    prompt.innerHTML = '<div style="text-align: center; color: var(--text-muted);">Validating Excel file...</div>';
    prompt.classList.add('show');
    
    // Validate Excel
    fetch('/api/pharmacy/reports/validate-excel', {
        method: 'POST',
        body: formData,
        headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            // Validation passed
            reportState.excelData = data.preview_rows;
            reportState.columnMapping = data.column_mapping;
            
            prompt.classList.remove('error');
            prompt.classList.add('success');
            prompt.innerHTML = `
                <div class="validation-prompt-title">✓ Excel Format Valid</div>
                <p style="margin: 0.5rem 0 0 0; color: var(--text-muted);">
                    Found ${data.total_rows} rows of data. Preview shown below.
                </p>
            `;
            
            // Show preview
            showExcelPreview(data.preview_rows, schema);
            
            // Update summary
            updateSummary(data.total_rows);
            
        } else {
            // Validation failed
            prompt.classList.remove('success');
            prompt.classList.add('error');
            prompt.innerHTML = `
                <div class="validation-prompt-title">❌ Invalid Excel Format</div>
                <p style="margin: 0.5rem 0 0 0; color: var(--text-muted);">${data.error}</p>
                <div class="validation-prompt-list">
                    <strong>Required Columns:</strong>
                    <ul>
                        ${data.required_columns.map(col => `<li>${col}</li>`).join('')}
                    </ul>
                </div>
                <p style="margin: 1rem 0 0 0; color: var(--text-muted);">Please update your file or download the correct template.</p>
            `;
            
            document.getElementById('previewSection').style.display = 'none';
        }
    })
    .catch(err => {
        prompt.classList.remove('success');
        prompt.classList.add('error');
        prompt.innerHTML = `<div class="validation-prompt-title">❌ Error</div><p>${err.message}</p>`;
    });
}

function showExcelPreview(rows, schema) {
    const previewSection = document.getElementById('previewSection');
    const previewTable = document.getElementById('previewTable');
    
    if (!rows || rows.length === 0) return;
    
    // Get headers from first row
    const headers = Object.keys(rows[0]);
    
    // Build table
    let html = '<tr>';
    headers.forEach(h => {
        html += `<th>${h}</th>`;
    });
    html += '</tr>';
    
    rows.forEach(row => {
        html += '<tr>';
        headers.forEach(h => {
            const value = row[h] || '-';
            html += `<td>${value}</td>`;
        });
        html += '</tr>';
    });
    
    previewTable.innerHTML = html;
    previewSection.style.display = 'block';
}

// ============================================================================
// FORM SUBMISSION
// ============================================================================

function collectFormData() {
    const records = [];
    const rows = document.querySelectorAll('.form-row');
    
    rows.forEach((row, rowIndex) => {
        const record = {};
        const inputs = row.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            const fieldName = input.name;
            let value = input.type === 'checkbox' ? input.checked : input.value;
            
            if (value !== '' && value !== false) {
                record[fieldName] = value;
            }
        });
        
        // Only add if has data
        if (Object.keys(record).length > 0) {
            records.push(record);
        }
    });
    
    return records;
}

function validateFormData(records) {
    const schema = SCHEMAS[reportState.reportType];
    const errors = [];
    
    records.forEach((record, idx) => {
        schema.forEach(field => {
            if (field.required && (!record[field.key] || record[field.key] === '')) {
                errors.push(`Record ${idx + 1}: ${field.label} is required`);
            }
        });
    });
    
    return errors;
}

function updateSummary(recordCount) {
    const summaryBox = document.getElementById('summaryBox');
    if (!summaryBox) return;
    
    document.getElementById('summaryReportType').textContent = 
        reportState.reportType.charAt(0).toUpperCase() + reportState.reportType.slice(1);
    
    document.getElementById('summaryEntryMode').textContent = 
        reportState.entryMode === 'manual' ? 'Manual Entry' : 'Excel Upload';
    
    document.getElementById('summaryRecordCount').textContent = recordCount;
    
    const scopeMap = {
        'anonymous': 'Anonymous (No identifiers)',
        'identified': 'Identified (Limited identifiers with consent)',
        'aggregated': 'Aggregated (Summary counts only)'
    };
    document.getElementById('summaryDataScope').textContent = scopeMap[reportState.reportType];
    
    summaryBox.style.display = 'block';
}

function submitReport() {
    let records = [];
    let recordCount = 0;
    
    if (reportState.entryMode === 'manual') {
        records = collectFormData();
        recordCount = records.length;
        
        const errors = validateFormData(records);
        if (errors.length > 0) {
            alert('Validation errors:\n' + errors.join('\n'));
            return;
        }
        
        if (recordCount === 0) {
            alert('Please enter at least one record');
            return;
        }
    } else {
        // Excel mode
        if (!reportState.excelData) {
            alert('Please upload and validate an Excel file first');
            return;
        }
        records = reportState.excelData;
        recordCount = records.length;
    }
    
    // Prepare submission
    const payload = {
        report_type: reportState.reportType,
        records: records
    };
    
    // Submit
    fetch('/api/pharmacy/reports/submit-manual', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            alert(`Success! Submitted ${recordCount} records.\nSubmission ID: ${data.submission_id}`);
            resetForm();
            // Redirect to history or dashboard
            setTimeout(() => {
                window.location.href = '/pharmacy/dashboard';
            }, 1500);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(err => {
        alert('Submission failed: ' + err.message);
    });
}

function resetForm() {
    reportState.records = [];
    reportState.excelData = null;
    reportState.columnMapping = null;
    
    document.getElementById('formFieldsContainer').innerHTML = '';
    document.getElementById('excelFile').value = '';
    document.getElementById('validationPrompt').classList.remove('show');
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('summaryBox').style.display = 'none';
    document.getElementById('confirmCheckbox').checked = false;
    document.getElementById('submitBtn').disabled = true;
    
    renderFormFields();
}

// Load XLSX library for Excel template download
const script = document.createElement('script');
script.src = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.min.js';
document.head.appendChild(script);
