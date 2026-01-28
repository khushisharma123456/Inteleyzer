// Safety Alerts Page JavaScript
lucide.createIcons();

let allAlerts = [];
let filteredAlerts = [];
let currentCategory = 'all';
let currentAlert = null;

// Fetch real alerts from database
async function loadAlerts() {
    try {
        console.log('Fetching alerts from /api/alerts...');
        const response = await fetch('/api/alerts');
        const data = await response.json();
        console.log('API Response:', data);
        const dbAlerts = data.alerts || [];
        console.log(`Loaded ${dbAlerts.length} alerts from database`);
        
        // Transform database alerts to match UI format
        allAlerts = dbAlerts.map(alert => ({
            id: `ALT-${alert.id}`,
            type: 'Safety Alert',
            drug: alert.drug_name,
            reason: alert.message,
            severity: alert.severity.toLowerCase(),
            source: alert.sender || 'System',
            timestamp: new Date(alert.created_at),
            category: 'all'
        }));
        
        filteredAlerts = allAlerts;
        renderAlerts();
    } catch (error) {
        console.error('Error loading alerts:', error);
    }
}

function renderAlerts() {
    const body = document.getElementById('alerts-body');

    if (filteredAlerts.length === 0) {
        body.innerHTML = '<tr><td colspan="6" class="empty-state"><i data-lucide="inbox"></i><p>All clear - no alerts at this time.</p></td></tr>';
        lucide.createIcons();
        return;
    }

    body.innerHTML = filteredAlerts.map(alert => {
        return `
        <tr onclick="openPanel('${alert.id}')">
            <td><strong>${alert.drug}</strong></td>
            <td>${alert.reason}</td>
            <td><span class="severity-badge severity-${alert.severity}">${alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}</span></td>
            <td>${alert.source}</td>
            <td>${alert.timestamp.toLocaleString()}</td>
            <td><button class="action-btn" onclick="event.stopPropagation(); openPanel('${alert.id}')">Review</button></td>
        </tr>
    `;
    }).join('');
    lucide.createIcons();
}

function filterByCategory(category) {
    currentCategory = category;
    if (category === 'all') {
        filteredAlerts = allAlerts;
    } else {
        filteredAlerts = allAlerts.filter(a => a.category === category);
    }
    renderAlerts();
}

function applySearch() {
    const query = document.getElementById('search-input').value.toLowerCase();
    if (!query) {
        filteredAlerts = allAlerts;
    } else {
        filteredAlerts = allAlerts.filter(alert =>
            alert.drug.toLowerCase().includes(query) ||
            alert.reason.toLowerCase().includes(query) ||
            alert.source.toLowerCase().includes(query)
        );
    }
    renderAlerts();
}

function openPanel(alertId) {
    currentAlert = allAlerts.find(a => a.id === alertId);
    if (!currentAlert) return;

    document.getElementById('detail-type').textContent = 'Safety Alert';
    document.getElementById('detail-type').className = `alert-type-badge type-interactions`;
    document.getElementById('detail-drug').textContent = currentAlert.drug;
    document.getElementById('detail-reason').textContent = currentAlert.reason;
    document.getElementById('detail-severity').textContent = currentAlert.severity.charAt(0).toUpperCase() + currentAlert.severity.slice(1);
    document.getElementById('detail-severity').className = `severity-badge severity-${currentAlert.severity}`;
    document.getElementById('detail-source').textContent = currentAlert.source;
    document.getElementById('detail-timestamp').textContent = currentAlert.timestamp.toLocaleString();

    // Hide request details section for safety alerts
    document.getElementById('request-details').style.display = 'none';

    populateActions();
    document.getElementById('sidePanel').classList.add('open');
}

function closePanel() {
    document.getElementById('sidePanel').classList.remove('open');
    currentAlert = null;
}

function populateActions() {
    const actionsDiv = document.getElementById('panel-actions');
    let html = `
        <button class="panel-btn btn-primary" onclick="takeAction('acknowledge')">Mark as Read</button>
        <button class="panel-btn btn-secondary" onclick="takeAction('defer')">Review Later</button>
        <button class="panel-btn btn-secondary" onclick="takeAction('close')">Close</button>
    `;

    actionsDiv.innerHTML = html;
}

window.takeAction = function(action) {
    let statusMessage = '';
    let showModal = true;

    switch(action) {
        case 'acknowledge':
            statusMessage = 'Alert has been acknowledged and marked as reviewed.';
            break;
        case 'defer':
            statusMessage = 'Alert has been deferred for later review.';
            break;
        case 'close':
            showModal = false;
            break;
    }

    if (showModal) {
        showActionModal(action, statusMessage);
    } else {
        closePanel();
    }
};

function showActionModal(action, statusMessage) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `
        <div class="modal-box">
            <div class="success-checkmark">
                <svg viewBox="0 0 24 24">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            </div>
            <h3>${action === 'acknowledge' ? 'Alert Acknowledged' : action === 'defer' ? 'Alert Deferred' : 'Alert Acknowledged'}</h3>
            <p style="color: var(--text-muted); margin: 0.5rem 0 0 0;">${currentAlert.drug}</p>
            <div class="modal-status">
                <strong>Status:</strong> ${statusMessage}
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    
    setTimeout(() => {
        overlay.remove();
        closePanel();
    }, 2000);
}

window.openPanel = openPanel;
window.closePanel = closePanel;
window.filterByCategory = filterByCategory;

// Load alerts from database on page load
loadAlerts();
