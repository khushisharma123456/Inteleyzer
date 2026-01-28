# Pharmacy Alerts - Acknowledge Status Fix

## Problem
When a pharmacy user clicked the "Acknowledge Alert" button, the status would update in the UI but would not persist to the database. Upon page refresh, the alert would revert to its original status.

## Root Cause
The backend endpoint `/api/pharmacy/alerts/<alert_id>/acknowledge` was not actually updating the database - it was just returning success without persisting the change.

## Solution Implemented

### 1. Enhanced Alert Model (models.py)
Added three new fields to the Alert model to track acknowledgment:

```python
status = db.Column(db.String(20), default='new')  # new, acknowledged, resolved
acknowledged_at = db.Column(db.DateTime, nullable=True)
acknowledged_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
```

Also added relationship to track which pharmacy acknowledged the alert:
```python
acknowledger = db.relationship('User', foreign_keys=[acknowledged_by], backref=db.backref('acknowledged_alerts', lazy=True))
```

Updated `to_dict()` method to include status and acknowledged_at in API responses.

### 2. Fixed Backend Endpoint (app.py)

**Old Implementation** (Non-functional):
```python
# Just returned success without updating database
return jsonify({
    'success': True,
    'message': 'Alert acknowledged',
    'acknowledged_at': datetime.datetime.now().isoformat()
})
```

**New Implementation** (Persistent):
```python
@app.route('/api/pharmacy/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_pharmacy_alert(alert_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        # Find the alert
        alert = Alert.query.get(alert_id)
        if not alert:
            return jsonify({'success': False, 'message': 'Alert not found'}), 404
        
        # Update alert status in database
        alert.status = 'acknowledged'
        alert.acknowledged_at = datetime.datetime.utcnow()
        alert.acknowledged_by = user.id
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Alert acknowledged',
            'status': alert.status,
            'acknowledged_at': alert.acknowledged_at.isoformat()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error acknowledging alert: {str(e)}'
        }), 500
```

### 3. Added Alerts Fetch Endpoint (app.py)

New endpoint to retrieve all alerts for a pharmacy:

```python
@app.route('/api/pharmacy/alerts', methods=['GET'])
def get_pharmacy_alerts():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'pharmacy':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        # Get all alerts for this pharmacy
        alerts = Alert.query.filter(
            Alert.recipient_type.in_(['all', 'pharmacy'])
        ).order_by(Alert.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'alerts': [alert.to_dict() for alert in alerts]
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching alerts: {str(e)}'
        }), 500
```

### 4. Updated Frontend (templates/pharmacy/alerts.html)

**Before**: Used hardcoded sample data
```javascript
const sampleAlerts = [
    { id: 'ALT-001', ... },
    { id: 'ALT-002', ... },
    // etc
];
allAlerts = sampleAlerts;
renderAlerts(allAlerts);
```

**After**: Fetches from backend
```javascript
async function loadAlerts() {
    try {
        const response = await fetch('/api/pharmacy/alerts');
        const data = await response.json();
        
        if (data.success && data.alerts) {
            allAlerts = data.alerts.map(alert => ({
                id: alert.id,
                type: alert.type || 'safety',
                title: alert.title || 'Alert',
                drug: alert.drug_name || 'Unknown',
                description: alert.message || '',
                reason: alert.reason || 'Safety monitoring',
                impact: alert.impact || 'Review recommended',
                severity: alert.severity ? alert.severity.toLowerCase() : 'medium',
                source: alert.sender || 'System Generated',
                status: alert.status || 'new',
                timestamp: new Date(alert.created_at),
                createdAt: alert.created_at
            }));
            
            filteredAlerts = allAlerts;
            renderAlerts(allAlerts);
        }
    } catch (error) {
        console.error('Error loading alerts:', error);
        loadSampleAlerts();
    }
}

// Initialize
loadAlerts();
```

### 5. Improved Acknowledge Handler

Enhanced error handling and feedback:

```javascript
window.acknowledgeAlert = async function(alertId) {
    try {
        const response = await fetch(`/api/pharmacy/alerts/${alertId}/acknowledge`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const result = await response.json();
        if (result.success) {
            // Update local state
            const alert = allAlerts.find(a => a.id === alertId);
            if (alert) {
                alert.status = 'acknowledged';
            }
            
            // Show success modal
            const modal = document.getElementById('success-modal');
            modal.classList.add('show');
            
            setTimeout(() => {
                modal.classList.remove('show');
                closePanel();
                renderAlerts(filteredAlerts);
            }, 2000);
        } else {
            alert('Error: ' + (result.message || 'Failed to acknowledge alert'));
        }
    } catch (error) {
        console.error('Error acknowledging alert:', error);
        alert('Error acknowledging alert');
    }
};
```

## Database Migration

A migration script was run to add the new columns to the Alert table:
- `status` (VARCHAR, default='new')
- `acknowledged_at` (DATETIME, nullable)
- `acknowledged_by` (INTEGER, foreign key to User)

## How It Works Now

1. **User clicks "Acknowledge Alert"**
   - Frontend sends POST request to `/api/pharmacy/alerts/{alertId}/acknowledge`

2. **Backend processes request**
   - Validates user is authenticated and is a pharmacy
   - Finds the alert in database
   - Updates alert status to 'acknowledged'
   - Records timestamp and pharmacy ID
   - Commits changes to database
   - Returns success response

3. **Frontend updates UI**
   - Updates local alert object status
   - Shows success modal
   - Closes detail panel
   - Re-renders alert table with updated status

4. **Status persists**
   - When page is refreshed, alerts are fetched from backend
   - Acknowledged alerts show status as "Acknowledged"
   - Status badge displays green color for acknowledged alerts

## Testing

### Test Case 1: Acknowledge Alert
1. Open Pharmacy Alerts page
2. Click "View" on a new alert
3. Click "Acknowledge Alert" button
4. See success modal
5. Alert status changes to "Acknowledged"
6. Refresh page
7. Alert still shows as "Acknowledged" ✅

### Test Case 2: Already Acknowledged Alert
1. Open alert that's already acknowledged
2. See message: "Status: Acknowledged - This alert has already been processed"
3. No acknowledge button shown ✅

### Test Case 3: Error Handling
1. Try to acknowledge with invalid alert ID
2. See error message: "Alert not found" ✅

## Files Modified

1. **models.py**
   - Added `status`, `acknowledged_at`, `acknowledged_by` fields to Alert model
   - Added `acknowledger` relationship
   - Updated `to_dict()` method

2. **app.py**
   - Added `/api/pharmacy/alerts` GET endpoint
   - Fixed `/api/pharmacy/alerts/<alert_id>/acknowledge` POST endpoint
   - Added proper database persistence

3. **templates/pharmacy/alerts.html**
   - Added `loadAlerts()` function to fetch from backend
   - Updated initialization to call `loadAlerts()`
   - Improved error handling in acknowledge function

## Benefits

✅ **Persistent Status**: Alert acknowledgment now persists across page refreshes
✅ **Audit Trail**: Records which pharmacy acknowledged and when
✅ **Better UX**: Users see accurate status immediately and after refresh
✅ **Error Handling**: Proper error messages if something goes wrong
✅ **Scalable**: Works with any number of alerts in database

## Future Enhancements

- Add ability to filter by status (New, Acknowledged, Resolved)
- Add bulk acknowledge functionality
- Add audit log view to see who acknowledged what and when
- Add ability to resolve alerts
- Add email notifications when alerts are acknowledged
