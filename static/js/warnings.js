// Drug Advisories Page JavaScript
lucide.createIcons();

let allWarnings = [];
let filteredWarnings = [];
let currentType = 'all';
let currentWarning = null;

// Fetch drug advisories from database
async function loadWarnings() {
    try {
        console.log('Fetching drug advisories from /api/drug-advisories...');
        const response = await fetch('/api/drug-advisories');
        const data = await response.json();
        console.log('API Response:', data);
        allWarnings = data.advisories || [];
        console.log(`Loaded ${allWarnings.length} drug advisories from database`);
        
        filteredWarnings = allWarnings;
        renderWarnings(filteredWarnings);
    } catch (error) {
        console.error('Error loading drug advisories:', error);
    }
}

function renderWarnings(warnings) {
    const body = document.getElementById('warnings-body');

    if (warnings.length === 0) {
        body.innerHTML = '<tr><td colspan="6" class="empty-state"><i data-lucide="inbox"></i><p>No active drug warnings at this time.</p></td></tr>';
        lucide.createIcons();
        return;
    }

    body.innerHTML = warnings.map(warning => `
        <tr onclick="openPanel('${warning.id}')">
            <td><span class="warning-type-badge type-${warning.type}">${warning.type.charAt(0).toUpperCase() + warning.type.slice(1)}</span></td>
            <td>${warning.drug}</td>
            <td>${warning.summary}</td>
            <td><span class="severity-badge severity-${warning.severity.toLowerCase()}">${warning.severity}</span></td>
            <td>${warning.source}</td>
            <td>${new Date(warning.date).toLocaleDateString()}</td>
        </tr>
    `).join('');

    lucide.createIcons();
}

function filterByType(type) {
    currentType = type;
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[onclick="filterByType('${type}')"]`).classList.add('active');

    if (type === 'all') {
        filteredWarnings = allWarnings;
    } else {
        filteredWarnings = allWarnings.filter(w => w.type === type);
    }
    renderWarnings(filteredWarnings);
}

function applySearch() {
    const query = document.getElementById('searchInput').value.toLowerCase();
    
    let results = currentType === 'all' ? allWarnings : allWarnings.filter(w => w.type === currentType);
    
    if (query) {
        results = results.filter(warning =>
            warning.drug.toLowerCase().includes(query) ||
            warning.summary.toLowerCase().includes(query) ||
            warning.description.toLowerCase().includes(query)
        );
    }
    
    filteredWarnings = results;
    renderWarnings(results);
}

function openPanel(warningId) {
    currentWarning = allWarnings.find(w => w.id == warningId);
    if (!currentWarning) return;

    document.getElementById('detail-type').textContent = currentWarning.type.charAt(0).toUpperCase() + currentWarning.type.slice(1);
    document.getElementById('detail-type').className = `warning-type-badge type-${currentWarning.type}`;
    document.getElementById('detail-drug').textContent = currentWarning.drug;
    document.getElementById('detail-severity').textContent = currentWarning.severity.charAt(0).toUpperCase() + currentWarning.severity.slice(1);
    document.getElementById('detail-severity').className = `severity-badge severity-${currentWarning.severity.toLowerCase()}`;
    document.getElementById('detail-description').textContent = currentWarning.description;
    document.getElementById('detail-source').textContent = currentWarning.source;
    document.getElementById('detail-date').textContent = new Date(currentWarning.date).toLocaleDateString();

    const refRow = document.getElementById('reference-row');
    if (currentWarning.reference) {
        refRow.style.display = 'block';
        document.getElementById('detail-reference').innerHTML = `<a href="${currentWarning.reference}" target="_blank">${currentWarning.reference}</a>`;
    } else {
        refRow.style.display = 'none';
    }

    document.getElementById('sidePanel').classList.add('open');
}

function closePanel() {
    document.getElementById('sidePanel').classList.remove('open');
    currentWarning = null;
}

window.filterByType = filterByType;
window.openPanel = openPanel;
window.closePanel = closePanel;

// Add event listener for search
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', applySearch);
    }
});

// Load drug advisories from database on page load
loadWarnings();
