// Report Submission History JavaScript
lucide.createIcons();

let submissionHistory = [];

// Fetch submission history from database
async function loadSubmissionHistory() {
    try {
        console.log('Fetching submission history from /api/side-effect-reports...');
        const response = await fetch('/api/side-effect-reports');
        const data = await response.json();
        console.log('API Response:', data);
        
        if (data.success) {
            submissionHistory = data.reports || [];
            console.log(`Loaded ${submissionHistory.length} reports from database`);
            renderSubmissionHistory();
        }
    } catch (error) {
        console.error('Error loading submission history:', error);
    }
}

function renderSubmissionHistory() {
    const tbody = document.getElementById('historyBody');
    
    if (submissionHistory.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 2rem; color: var(--text-muted);">No submissions yet</td></tr>';
        return;
    }
    
    tbody.innerHTML = submissionHistory.map(report => `
        <tr>
            <td><span class="type-badge type-${report.reportType === 'patient-linked' ? 'linked' : 'anonymised'}">${report.reportType === 'patient-linked' ? 'Patient-linked' : 'Anonymised'}</span></td>
            <td>${report.drugName}</td>
            <td>${new Date(report.dateSubmitted).toLocaleDateString()}</td>
            <td><span class="status-badge status-submitted">${report.status}</span></td>
        </tr>
    `).join('');
}

// Submit new report
async function submitReportToDatabase(reportType, patientId = null) {
    const drugName = document.querySelector('input[placeholder="e.g., Aspirin, Metformin"]').value;
    const sideEffect = document.querySelector('textarea[placeholder="Describe what you observed..."]').value;
    const severity = document.getElementById('severitySelect')?.value || 'Medium';
    
    console.log('Submitting report:', { drugName, sideEffect, severity, reportType, patientId });
    
    try {
        const response = await fetch('/api/report-side-effect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                drug_name: drugName,
                side_effect: sideEffect,
                severity: severity,
                patient_id: patientId,
                report_type: reportType
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            console.log('Report submitted successfully:', data);
            // Reload submission history
            await loadSubmissionHistory();
            return true;
        } else {
            console.error('Failed to submit report:', data.message);
            alert('Error: ' + (data.message || 'Failed to submit report'));
            return false;
        }
    } catch (error) {
        console.error('Error submitting report:', error);
        alert('Error submitting report: ' + error.message);
        return false;
    }
}

// Load history when page loads
window.addEventListener('DOMContentLoaded', function() {
    loadSubmissionHistory();
});

// Export functions for use in inline scripts
window.loadSubmissionHistory = loadSubmissionHistory;
window.submitReportToDatabase = submitReportToDatabase;
