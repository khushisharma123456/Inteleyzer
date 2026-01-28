# Pharmacy Alerts Module - Firebase Setup Guide

## Overview
The Pharmacy Alerts module uses Firebase Firestore for real-time alert management and Firebase Authentication for security. This document outlines the data structure, security rules, and implementation details.

---

## 1. Firebase Collections Structure

### Collection: `alerts`
Stores all safety alerts, compliance notifications, and follow-up alerts.

```javascript
{
  alert_id: "ALT-001",                    // Unique identifier
  pharmacy_id: "PH-CVS-001",              // Pharmacy receiving the alert
  alert_type: "safety",                   // "safety" | "followup" | "compliance"
  severity: "critical",                   // "low" | "medium" | "high" | "critical"
  drug_name: "PainAway (Ibuprofen 500mg)",
  title: "Drug Batch Recall",
  description: "Batch PW-2024-001 has been recalled due to contamination risk.",
  reason: "Manufacturing quality control failure detected",
  impact: "Immediate discontinuation recommended.",
  source: "Pharma Company",               // "Pharma Company" | "System Generated"
  status: "new",                          // "new" | "acknowledged" | "resolved"
  created_at: Timestamp,
  created_by: "system",                   // User/system that created alert
  acknowledged_by: null,                  // pharmacy_id when acknowledged
  acknowledged_at: null,                  // Timestamp when acknowledged
  archived: false
}
```

### Collection: `alert_activity_logs`
Audit trail for all alert actions.

```javascript
{
  log_id: "LOG-001",
  alert_id: "ALT-001",
  pharmacy_id: "PH-CVS-001",
  action: "acknowledged",                 // "acknowledged" | "viewed" | "archived"
  timestamp: Timestamp,
  user_id: "pharmacy_user_123"
}
```

---

## 2. Firebase Security Rules

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    
    // Alerts collection - Pharmacy can only read their own alerts
    match /alerts/{document=**} {
      allow read: if request.auth != null && 
                     (resource.data.pharmacy_id == request.auth.uid || 
                      request.auth.token.role == 'admin');
      
      allow update: if request.auth != null && 
                       resource.data.pharmacy_id == request.auth.uid &&
                       request.resource.data.status in ['acknowledged', 'resolved'] &&
                       request.resource.data.acknowledged_by == request.auth.uid;
      
      allow create: if request.auth.token.role == 'admin' || 
                       request.auth.token.role == 'pharma_company';
      
      allow delete: if request.auth.token.role == 'admin';
    }
    
    // Alert activity logs - Pharmacy can only read their own logs
    match /alert_activity_logs/{document=**} {
      allow read: if request.auth != null && 
                     (resource.data.pharmacy_id == request.auth.uid || 
                      request.auth.token.role == 'admin');
      
      allow create: if request.auth != null && 
                       request.resource.data.pharmacy_id == request.auth.uid;
      
      allow delete: if request.auth.token.role == 'admin';
    }
  }
}
```

---

## 3. Alert Types & Categories

### Safety Alerts
- **Purpose**: Drug-level risk notifications
- **Examples**:
  - Batch recalls
  - Adverse event signals
  - Safety advisories
  - Severity escalations
- **Pharmacy Action**: Review and acknowledge

### Follow-up Required
- **Purpose**: Informational alerts for clinical follow-up
- **Examples**:
  - Case flagged for clinical investigation
  - Unusual patient response patterns
- **Pharmacy Action**: None (informational only)
- **UI Behavior**: Disable action buttons, show "No action required" message

### Compliance Alerts
- **Purpose**: Operational compliance notifications
- **Examples**:
  - Missing reports
  - Delayed submissions
  - Policy updates
- **Pharmacy Action**: Review and acknowledge

---

## 4. Alert Severity Levels

| Severity | Color | Meaning | Response Time |
|----------|-------|---------|----------------|
| Low | Green | Minor issue, informational | 7 days |
| Medium | Yellow | Moderate concern, review recommended | 3 days |
| High | Red | Serious issue, immediate review needed | 24 hours |
| Critical | Dark Red | Emergency, immediate action required | 1 hour |

---

## 5. Alert Status Lifecycle

```
new → acknowledged → resolved
  ↓
  └─→ archived (after 90 days)
```

**Status Definitions:**
- **new**: Alert created, not yet reviewed by pharmacy
- **acknowledged**: Pharmacy has reviewed and confirmed receipt
- **resolved**: Alert has been addressed (manual status update by admin)
- **archived**: Alert older than 90 days, moved to archive

---

## 6. Implementation: Acknowledge Flow

### Frontend (JavaScript)
```javascript
async function acknowledgeAlert(alertId) {
  const response = await fetch(`/api/pharmacy/alerts/${alertId}/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  
  const result = await response.json();
  if (result.success) {
    // Update UI
    showSuccessModal('Alert acknowledged');
    updateAlertStatus(alertId, 'acknowledged');
  }
}
```

### Backend (Flask → Firebase)
```python
@app.route('/api/pharmacy/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_pharmacy_alert(alert_id):
    if 'user_id' not in session:
        return jsonify({'success': False}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False}), 403
    
    # Update Firestore
    db.collection('alerts').document(alert_id).update({
        'status': 'acknowledged',
        'acknowledged_by': user.id,
        'acknowledged_at': firestore.SERVER_TIMESTAMP
    })
    
    # Log action
    db.collection('alert_activity_logs').add({
        'alert_id': alert_id,
        'pharmacy_id': user.id,
        'action': 'acknowledged',
        'timestamp': firestore.SERVER_TIMESTAMP,
        'user_id': user.id
    })
    
    return jsonify({'success': True})
```

---

## 7. Data Privacy & Security

### ⚠️ CRITICAL: No Patient Identity
- Alerts NEVER contain patient names, ages, or identifiers
- Alerts NEVER contain patient contact information
- Alerts are drug-level or system-level notifications only

### Pharmacy Isolation
- Each pharmacy can only see alerts linked to their `pharmacy_id`
- Pharmacies cannot:
  - Edit alert content
  - Create new alerts
  - View other pharmacies' alerts
  - Contact patients

### Audit Trail
- All alert actions logged in `alert_activity_logs`
- Timestamps recorded for compliance
- User ID tracked for accountability

---

## 8. Empty State Handling

**When no alerts exist:**
```
"No active alerts at this time."
```

**Never show:**
- Empty tables
- Blank charts
- Loading spinners indefinitely

---

## 9. Firebase Initialization (Python)

```python
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Query pharmacy alerts
def get_pharmacy_alerts(pharmacy_id):
    alerts = db.collection('alerts').where(
        'pharmacy_id', '==', pharmacy_id
    ).where(
        'archived', '==', False
    ).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    
    return [doc.to_dict() for doc in alerts]
```

---

## 10. Testing Checklist

- [ ] Pharmacy can view only their own alerts
- [ ] Pharmacy cannot view other pharmacies' alerts
- [ ] Acknowledge button updates status to "acknowledged"
- [ ] Follow-up alerts show "No action required" message
- [ ] Action buttons disabled for follow-up alerts
- [ ] Acknowledge action logged in activity logs
- [ ] Empty state displays when no alerts exist
- [ ] Search filters work correctly
- [ ] Alert detail panel opens on row click
- [ ] Close button closes detail panel
- [ ] Success modal shows after acknowledge
- [ ] No patient identity visible in any alert

---

## 11. Compliance & Auditability

**For Regulatory Compliance:**
- All alert acknowledgments timestamped
- User ID recorded for each action
- Activity logs retained for 7 years
- Alerts cannot be deleted by pharmacies
- Audit trail immutable

**Reporting:**
```javascript
// Generate compliance report
db.collection('alert_activity_logs')
  .where('pharmacy_id', '==', pharmacyId)
  .where('timestamp', '>=', startDate)
  .where('timestamp', '<=', endDate)
  .orderBy('timestamp')
  .get()
```

---

## 12. Future Enhancements

- [ ] Real-time alert notifications via Firebase Cloud Messaging
- [ ] Batch alert acknowledgment
- [ ] Alert filtering by date range
- [ ] Export alert history as PDF
- [ ] Integration with pharmacy management systems
- [ ] Automated compliance reporting

---

## Support & Questions

For implementation questions or issues, refer to:
- Firebase Documentation: https://firebase.google.com/docs
- Firestore Security Rules: https://firebase.google.com/docs/firestore/security/start
- Python Firebase Admin SDK: https://firebase.google.com/docs/database/admin/start
