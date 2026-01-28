/**
 * Pharmacy Reports Module - Production-Ready
 * Compliance-grade UX with dialogs, confirmations, and proper validation
 */

let reportState = {
    reportType: 'anonymous',
    entryMode: 'manual',
    records: [],
    schema: null,
    excelData: null,
    consentGiven: false
};

let confirmAction = null;

const SCHEMAS = {
    anonymous: [
        { key: 'drug_name', label: 'Drug Name', type: 'text', required: true, placeholder: 'e.g., Aspirin 500mg' },
        { key: 'batch_lot_number', label: 'Batch / Lot Number', type: 'text', required: false },
        { key: 'dosage_form', label: 'Dosage Form', type: 'text', required: true },
        { key: 'date_of_dispensing', label: 'Date of Dispensing', type: 'date', required: true },
        { key: 'reaction_category', label: 'Reaction Category', type: 'text', required: true },
        { key: 'severity', label: 'Severity', type: 'select', options: ['mild', 'moderate', 'severe'], required: true, help: 'Select the severity level of the reaction' },
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
        { key: 'severity', label: 'Severity', type: 'select', options: ['mild', 'moderate', 'severe'], required: true, help: 'Select the severity level of the reaction' },
        { key: 'reaction_outcome', label: 'Reaction Outcome', type: 'select', options: ['recovered', 'recovering', 'not_recovered', 'fatal', 'unknown'], required: false },
        { key: 'age_group', label: 'Age Group', type: 'select', options: ['pediatric', 'adolescent', 'adult', 'elderly', 'unknown'], required: true },
        { key: 'gender', label: 'Gender', type: 'select', options: ['male', 'female', 'other', 'not_specified'], required: false },
        { key: 'internal_case_id', label: 'Internal Case ID', type: 'text', required: false },
        { key: 'treating_hospital_reference', label: 'Treating Hospital / Doctor Reference', type: 'text', required: false },
        { key: 'treating_doctor_name', label: 'Treating Doctor Name', type: 'text', required: false },
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

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    renderFormFields();
});

function initializeEventListeners() {
    document.querySelectorAll('input[name="report_type"]').forEach(radio => {
        radio.addEventListener('change', function() {
            reportState.reportType = this.value;
            reportState.consentGiven = false;
            updateReportTypeUI();
            renderFormFields();
            clearErrors();
        });
    });
    
    document.querySelectorAll('input[name="entry_mode"]').forEach(radio => {
        radio.addEventListener('change', function() {
            reportState.entryMode = this.value;
            updateEntryModeUI();
            clearErrors();
        });
    });
    
    const consentCheckbox = document.getElementById('consentCheckbox');
    if (consentCheckbox) {
        consentCheckbox.addEventListener('change', function() {
            reportState.consentGiven = this.checked;
            updateFormDisabledState();
            clearErrors();
        });
    }
    
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
    
    const confirmCheckbox = document.getElementById('confirmCheckbox');
    if (confirmCheckbox) {
        confirmCheckbox.addEventListener('change', function() {
            updateSubmitButtonState();
        });
    }
}

function updateReportTypeUI() {
    const warning = document.getElementById('identifiedWarning');
    const consentSection = document.getElementById('consentSection');
    
    if (reportState.reportType === 'identified') {
        warning.classList.add('show');
        consentSection.style.display = 'flex';
    } else {
        warning.classList.remove('show');
        consentSection.style.display = 'none';
    }
}

function updateEntryModeUI() {
    const manualSection = document.getElementById('manualEntrySection');
    const excelSection = document.getElementById('excelUploadSection');
    
    if (reportState.entryMode === 'manual') {
        manualSection.style.display = 'block';
        excelSection.style.display = 'none';
        document.getElementById('summaryBox').classList.add('show');
    } else {
        manualSection.style.display = 'none';
        excelSection.style.display = 'block';
        document.getElementById('summaryBox').classList.remove('show');
    }
}

function updateFormDisabledState() {
    const container = document.getElementById('formFieldsContainer');
    if (!container) return;
    
    const isDisabled = reportState.reportType === 'identified' && !reportState.consentGiven;
    const inputs = container.querySelectorAll('input, select, textarea, button');
    
    inputs.forEach(input => {
        if (isDisabled) {
            input.disabled = true;
            input.style.opacity = '0.5';
            input.style.pointerEvents = 'none';
        } else {
            input.disabled = false;
            input.style.opacity = '1';
            input.style.pointerEvents = 'auto';
        }
    });
}

function renderFormFields() {
    const container = document.getElementById('formFieldsContainer');
    if (!container) return;
    
    container.innerHTML = '';
    const schema = SCHEMAS[reportState.reportType];
    reportState.schema = schema;
    
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
            ${rowIndex > 0 ? `<button type="button" class="form-row-remove" onclick="showRemoveRecordDialog(${rowIndex})">Remove</button>` : ''}
        </div>
        <div class="form-grid">
    `;
    
    schema.forEach((field, idx) => {
        if (idx % 2 === 0) html += '<div>';
        html += renderFormField(field, rowIndex);
        if (idx % 2 === 1 || idx === schema.length - 1) html += '</div>';
    });
    
    html += '</div>';
    rowDiv.innerHTML = html;
    container.appendChild(rowDiv);
    
    // Animate new record
    rowDiv.style.animation = 'slideUp 0.3s ease';
    rowDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    updateFormDisabledState();
}

function showRemoveRecordDialog(rowIndex) {
    confirmAction = () => removeFormRow(rowIndex);
    
    const dialog = document.getElementById('confirmDialog');
    document.getElementById('confirmTitle').textContent = 'Remove Record';
    document.getElementById('confirmMessage').textContent = 'Are you sure you want to remove this record?';
    document.getElementById('confirmBtn').textContent = 'Remove';
    document.getElementById('confirmBtn').className = 'modal-btn modal-btn-confirm';
    
    dialog.classList.add('show');
}

function removeFormRow(rowIndex) {
    const container = document.getElementById('formFieldsContainer');
    const rows = container.querySelectorAll('.form-row');
    
    if (rows.length === 1) {
        showToast('At least one record must exist', 'error');
        return;
    }
    
    if (rows[rowIndex]) {
        rows[rowIndex].remove();
        showToast('Record removed', 'success');
        
        // Re-number remaining records
        container.querySelectorAll('.form-row').forEach((row, idx) => {
            const title = row.querySelector('.form-row-title');
            if (title) title.textContent = `Record ${idx + 1}`;
            
            const removeBtn = row.querySelector('.form-row-remove');
            if (removeBtn) {
                if (idx === 0) {
                    removeBtn.style.display = 'none';
                } else {
                    removeBtn.style.display = 'block';
                    removeBtn.onclick = () => showRemoveRecordDialog(idx);
                }
            }
        });
    }
    
    closeConfirmDialog();
}

function renderFormField(field, rowIndex) {
    const fieldId = `${field.key}_${rowIndex}`;
    const required = field.required ? '<span class="required">*</span>' : '';
    
    let input = '';
    
    switch (field.type) {
        case 'text':
        case 'number':
            input = `<input type="${field.type}" id="${fieldId}" name="${field.key}" placeholder="${field.placeholder || ''}" ${field.required ? 'required' : ''}>`;
            break;
        
        case 'date':
            input = `<input type="date" id="${fieldId}" name="${field.key}" ${field.required ? 'required' : ''}>`;
            break;
        
        case 'select':
            input = `<select id="${fieldId}" name="${field.key}" ${field.required ? 'required' : ''}>
                <option value="">-- Select --</option>
                ${field.options.map(opt => `<option value="${opt}">${opt.charAt(0).toUpperCase() + opt.slice(1).replace(/_/g, ' ')}</option>`).join('')}
            </select>`;
            break;
        
        case 'textarea':
            input = `<textarea id="${fieldId}" name="${field.key}" placeholder="${field.placeholder || ''}" ${field.required ? 'required' : ''} style="min-height: 60px;"></textarea>`;
            break;
    }
    
    return `
        <div class="form-group">
            <label for="${fieldId}">${field.label} ${required}</label>
            ${input}
            <span class="field-error"></span>
        </div>
    `;
}

function downloadTemplate() {
    const reportType = reportState.reportType;
    const schema = SCHEMAS[reportType];
    
    try {
        const wb = XLSX.utils.book_new();
        const headers = schema.map(f => f.label);
        const descRow = schema.map(f => f.required ? 'Required' : 'Optional');
        
        const ws = XLSX.utils.aoa_to_sheet([headers, descRow]);
        ws['!cols'] = schema.map(() => ({ wch: 18 }));
        
        XLSX.utils.book_append_sheet(wb, ws, 'Data');
        XLSX.writeFile(wb, `pharmacy_report_${reportType}.xlsx`);
        
        showToast('Template downloaded successfully', 'success');
    } catch (err) {
        showToast('Template download failed. Please try again.', 'error');
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = new Uint8Array(e.target.result);
            const workbook = XLSX.read(data, { type: 'array' });
            const worksheet = workbook.Sheets[workbook.SheetNames[0]];
            const rows = XLSX.utils.sheet_to_json(worksheet);
            
            if (rows.length === 0) {
                showExcelError('Excel file is empty');
                return;
            }
            
            const schema = SCHEMAS[reportState.reportType];
            const requiredColumns = schema.map(f => f.label);
            const fileColumns = Object.keys(rows[0] || {});
            
            // Check for missing columns
            const missingColumns = requiredColumns.filter(col => !fileColumns.includes(col));
            
            // Check for extra columns
            const extraColumns = fileColumns.filter(col => !requiredColumns.includes(col));
            
            const prompt = document.getElementById('validationPrompt');
            
            if (missingColumns.length > 0 || extraColumns.length > 0) {
                let errorMsg = '❌ Invalid Excel Format\n\nThe uploaded file does not match the required structure.\n\n';
                if (missingColumns.length > 0) {
                    errorMsg += `Missing columns: ${missingColumns.join(', ')}\n\n`;
                }
                if (extraColumns.length > 0) {
                    errorMsg += `Extra columns (not allowed): ${extraColumns.join(', ')}\n\n`;
                }
                errorMsg += 'Required Columns:\n' + requiredColumns.join('\n') + '\n\nPlease update the file or download the correct template.';
                
                showExcelError(errorMsg);
                document.getElementById('previewSection').style.display = 'none';
                document.getElementById('submitBtn').disabled = true;
            } else {
                reportState.excelData = rows;
                
                prompt.classList.remove('error');
                prompt.classList.add('success');
                prompt.innerHTML = `
                    <div class="validation-prompt-title">✓ Valid Format</div>
                    <p style="margin: 0.5rem 0 0 0; color: var(--text-muted); font-size: 0.85rem;">${rows.length} records found</p>
                `;
                
                showExcelPreview(rows);
                updateSummary(rows.length);
                document.getElementById('submitBtn').disabled = false;
                
                prompt.classList.add('show');
                showToast(`Excel file validated: ${rows.length} records`, 'success');
            }
        } catch (err) {
            showExcelError('Error reading Excel file: ' + err.message);
        }
    };
    
    reader.readAsArrayBuffer(file);
}

function showExcelError(message) {
    const prompt = document.getElementById('validationPrompt');
    prompt.classList.remove('success');
    prompt.classList.add('error');
    prompt.innerHTML = `<div class="validation-prompt-title">${message}</div>`;
    prompt.classList.add('show');
    showToast('Excel validation failed', 'error');
}

function showExcelPreview(rows) {
    const previewSection = document.getElementById('previewSection');
    const previewTable = document.getElementById('previewTable');
    
    if (!rows || rows.length === 0) return;
    
    const headers = Object.keys(rows[0]);
    
    let html = '<tr>';
    headers.forEach(h => {
        html += `<th>${h}</th>`;
    });
    html += '</tr>';
    
    rows.slice(0, 5).forEach(row => {
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

function collectFormData() {
    const records = [];
    const rows = document.querySelectorAll('.form-row');
    
    rows.forEach((row) => {
        const record = {};
        const inputs = row.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            const fieldName = input.name;
            let value = input.type === 'checkbox' ? input.checked : input.value;
            
            if (value !== '' && value !== false) {
                record[fieldName] = value;
            }
        });
        
        if (Object.keys(record).length > 0) {
            records.push(record);
        }
    });
    
    return records;
}

function validateFormFields() {
    const container = document.getElementById('formFieldsContainer');
    const rows = container.querySelectorAll('.form-row');
    const errors = [];
    let firstInvalidField = null;
    
    rows.forEach((row, rowIdx) => {
        const inputs = row.querySelectorAll('input, select, textarea');
        const schema = SCHEMAS[reportState.reportType];
        
        inputs.forEach(input => {
            const fieldName = input.name;
            const field = schema.find(f => f.key === fieldName);
            
            if (!field) return;
            
            const formGroup = input.closest('.form-group');
            const value = input.type === 'checkbox' ? input.checked : input.value;
            const isEmpty = !value || value === '';
            
            if (field.required && isEmpty) {
                formGroup.classList.add('error');
                const errorSpan = formGroup.querySelector('.field-error');
                if (errorSpan) {
                    errorSpan.textContent = 'This field is required';
                }
                errors.push(`Record ${rowIdx + 1}: ${field.label}`);
                
                if (!firstInvalidField) {
                    firstInvalidField = input;
                }
            } else {
                formGroup.classList.remove('error');
                const errorSpan = formGroup.querySelector('.field-error');
                if (errorSpan) {
                    errorSpan.textContent = '';
                }
            }
        });
    });
    
    if (firstInvalidField) {
        firstInvalidField.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    return errors;
}

function showErrorSummary(errors) {
    const errorSummary = document.getElementById('errorSummary');
    const errorList = document.getElementById('errorList');
    
    errorList.innerHTML = errors.map(err => `<li>${err}</li>`).join('');
    errorSummary.classList.add('show');
    
    errorSummary.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearErrors() {
    const errorSummary = document.getElementById('errorSummary');
    errorSummary.classList.remove('show');
    
    const formGroups = document.querySelectorAll('.form-group.error');
    formGroups.forEach(group => {
        group.classList.remove('error');
        const errorSpan = group.querySelector('.field-error');
        if (errorSpan) {
            errorSpan.textContent = '';
        }
    });
    
    const consentError = document.getElementById('consentError');
    if (consentError) {
        consentError.classList.remove('show');
    }
}

function updateSummary(recordCount) {
    const summaryBox = document.getElementById('summaryBox');
    if (!summaryBox) return;
    
    document.getElementById('summaryReportType').textContent = 
        reportState.reportType.charAt(0).toUpperCase() + reportState.reportType.slice(1).replace(/_/g, ' ');
    
    document.getElementById('summaryEntryMode').textContent = 
        reportState.entryMode === 'manual' ? 'Manual Entry' : 'Excel Upload';
    
    document.getElementById('summaryRecordCount').textContent = recordCount;
    
    summaryBox.classList.add('show');
}

function updateSubmitButtonState() {
    const submitBtn = document.getElementById('submitBtn');
    const confirmCheckbox = document.getElementById('confirmCheckbox');
    
    submitBtn.disabled = !confirmCheckbox.checked;
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastText = document.getElementById('toastText');
    
    toast.className = `toast ${type === 'error' ? 'toast-error' : 'toast-success'}`;
    toastText.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function showClearDialog() {
    document.getElementById('clearDialog').classList.add('show');
}

function closeClearDialog() {
    document.getElementById('clearDialog').classList.remove('show');
}

function confirmClearForm() {
    closeClearDialog();
    resetForm();
    showToast('Form cleared', 'success');
}

function closeConfirmDialog() {
    document.getElementById('confirmDialog').classList.remove('show');
    confirmAction = null;
}

function executeConfirmAction() {
    if (confirmAction) {
        confirmAction();
    }
    closeConfirmDialog();
}

function submitReport() {
    clearErrors();
    
    // Check consent for identified data
    if (reportState.reportType === 'identified' && !reportState.consentGiven) {
        const consentError = document.getElementById('consentError');
        consentError.classList.add('show');
        document.getElementById('consentSection').scrollIntoView({ behavior: 'smooth', block: 'center' });
        showToast('Consent is required to submit identifiable data', 'error');
        return;
    }
    
    let records = [];
    let recordCount = 0;
    
    if (reportState.entryMode === 'manual') {
        records = collectFormData();
        recordCount = records.length;
        
        // Validate form fields
        const fieldErrors = validateFormFields();
        if (fieldErrors.length > 0) {
            showErrorSummary(fieldErrors);
            showToast('Please complete all required fields', 'error');
            return;
        }
        
        if (recordCount === 0) {
            showToast('Please enter at least one record', 'error');
            return;
        }
    } else {
        if (!reportState.excelData) {
            showToast('Please upload and validate an Excel file first', 'error');
            return;
        }
        records = reportState.excelData;
        recordCount = records.length;
    }
    
    const payload = {
        report_type: reportState.reportType,
        entry_mode: reportState.entryMode,
        records: records
    };
    
    fetch('/api/pharmacy/reports/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            showSuccessMessage(recordCount);
            setTimeout(() => {
                resetForm();
            }, 1500);
        } else {
            showToast(data.message || 'Submission failed', 'error');
        }
    })
    .catch(err => {
        showToast('Error: ' + err.message, 'error');
    });
}

function showSuccessMessage(recordCount) {
    const successMessage = document.getElementById('successMessage');
    successMessage.classList.add('show');
    successMessage.scrollIntoView({ behavior: 'smooth', block: 'start' });
    showToast(`✅ ${recordCount} record(s) submitted successfully`, 'success');
}

function resetForm() {
    reportState.records = [];
    reportState.excelData = null;
    reportState.consentGiven = false;
    
    const container = document.getElementById('formFieldsContainer');
    if (container) container.innerHTML = '';
    
    const fileInput = document.getElementById('excelFile');
    if (fileInput) fileInput.value = '';
    
    const validationPrompt = document.getElementById('validationPrompt');
    if (validationPrompt) validationPrompt.classList.remove('show');
    
    const previewSection = document.getElementById('previewSection');
    if (previewSection) previewSection.style.display = 'none';
    
    const summaryBox = document.getElementById('summaryBox');
    if (summaryBox) summaryBox.classList.remove('show');
    
    const confirmCheckbox = document.getElementById('confirmCheckbox');
    if (confirmCheckbox) confirmCheckbox.checked = false;
    
    const consentCheckbox = document.getElementById('consentCheckbox');
    if (consentCheckbox) consentCheckbox.checked = false;
    
    const successMessage = document.getElementById('successMessage');
    if (successMessage) successMessage.classList.remove('show');
    
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) submitBtn.disabled = true;
    
    clearErrors();
    
    // Reset to anonymous and manual
    document.getElementById('type-anon').checked = true;
    document.getElementById('mode-manual').checked = true;
    reportState.reportType = 'anonymous';
    reportState.entryMode = 'manual';
    
    updateReportTypeUI();
    updateEntryModeUI();
    renderFormFields();
}

// Load XLSX library
const script = document.createElement('script');
script.src = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.min.js';
document.head.appendChild(script);
