# JWT & Session Expiry - Quick Reference

## ⚡ Quick Start Testing

### 1. Default Configuration
- **JWT Token Expiry**: 24 hours
- **Session Timeout**: 30 minutes of inactivity
- **Token Refresh**: Automatic before expiry

### 2. For Quick Testing (5-minute test)

**Edit `auth_config.py`**:
```python
SESSION_TIMEOUT_MINUTES = 1       # 1 minute (instead of 30)
TOKEN_EXPIRY_HOURS = 0.083        # 5 minutes (instead of 24)
```

**Restart app**:
```bash
# Stop: Ctrl+C
python app.py
```

### 3. Test Steps (5 minutes total)

1. **Login** → http://127.0.0.1:5000/login
   - Email: `doctor@hospital.com`
   - Password: `password123`

2. **Verify token** (in browser console F12):
   ```javascript
   localStorage.getItem('jwt_token')  // Should show token
   ```

3. **Wait 1 minute** (session timeout)

4. **Try to access dashboard** → Should redirect to login

5. **Check console**:
   ```javascript
   localStorage.getItem('jwt_token')  // Should be null (cleared)
   ```

---

## 📊 Testing Timeline

### Quick Test (5 minutes)
```
0:00 - Login
0:30 - Verify token in console
1:00 - Wait for session timeout
1:30 - Try to access page → Redirected to login ✅
```

### Medium Test (35 minutes)
```
0:00 - Login (set SESSION_TIMEOUT_MINUTES = 30)
5:00 - Verify token exists
15:00 - Click on page (refresh session)
30:00 - Wait without activity
31:00 - Try to access page → Redirected to login ✅
```

### Full Test (24+ hours)
```
0:00 - Login (set TOKEN_EXPIRY_HOURS = 24)
12:00 - Token still valid
23:00 - Token still valid
24:00 - Token expired → Redirect to login ✅
```

---

## 🔍 Browser Console Commands

### Check Token Status
```javascript
// Token exists?
localStorage.getItem('jwt_token') ? '✅ Token exists' : '❌ No token'

// How long until token expires?
const expiry = parseInt(localStorage.getItem('token_expiry'));
const now = new Date().getTime();
const minutesLeft = Math.round((expiry - now) / 60000);
console.log(`⏱️ Token expires in ${minutesLeft} minutes`);

// How long until session expires?
const sessionStart = parseInt(localStorage.getItem('session_start'));
const sessionTimeout = parseInt(localStorage.getItem('session_timeout'));
const sessionAge = now - sessionStart;
const minutesUntilExpiry = Math.round((sessionTimeout - sessionAge) / 60000);
console.log(`⏱️ Session expires in ${minutesUntilExpiry} minutes`);
```

### Force Token Expiry (for testing)
```javascript
// Make token expire immediately
localStorage.setItem('token_expiry', new Date().getTime());
console.log('✅ Token set to expire now');

// Try to access page - should redirect to login
window.location.href = '/doctor/dashboard';
```

### Force Session Expiry (for testing)
```javascript
// Make session expire immediately (31 minutes ago with 30-min timeout)
const thirtyOneMinutesAgo = new Date().getTime() - (31 * 60 * 1000);
localStorage.setItem('session_start', thirtyOneMinutesAgo);
console.log('✅ Session set to expire now');

// Try to access page - should redirect to login
window.location.href = '/doctor/dashboard';
```

---

## 🔐 API Testing

### Login and Get Token
```bash
curl -X POST http://127.0.0.1:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"doctor@hospital.com","password":"password123"}'
```

**Response**:
```json
{
  "success": true,
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_expiry": 24,
  "session_timeout": 30
}
```

### Use Token in API Call
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."

curl -X GET http://127.0.0.1:5000/api/patients \
  -H "Authorization: Bearer $TOKEN"
```

### Refresh Token
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."

curl -X POST http://127.0.0.1:5000/api/auth/refresh-token \
  -H "Authorization: Bearer $TOKEN"
```

---

## 📋 Configuration Options

### File: `auth_config.py`

```python
# Session timeout (minutes)
SESSION_TIMEOUT_MINUTES = 30      # Default: 30 minutes

# Token expiry (hours)
TOKEN_EXPIRY_HOURS = 24           # Default: 24 hours

# JWT Secret (change in production!)
JWT_SECRET = 'your-secret-key-change-in-production'
```

### Quick Test Settings
```python
SESSION_TIMEOUT_MINUTES = 1       # 1 minute
TOKEN_EXPIRY_HOURS = 0.083        # 5 minutes
```

### Production Settings
```python
SESSION_TIMEOUT_MINUTES = 60      # 1 hour
TOKEN_EXPIRY_HOURS = 24           # 24 hours
JWT_SECRET = os.environ.get('JWT_SECRET')  # From env var
```

---

## ✅ Testing Checklist

### Basic (2 minutes)
- [ ] Login works
- [ ] Token is generated
- [ ] Token stored in localStorage
- [ ] Logout clears token

### Session (35 minutes)
- [ ] Set SESSION_TIMEOUT_MINUTES = 1
- [ ] Login
- [ ] Wait 1 minute
- [ ] Try to access page → Redirected to login

### Token (5 minutes)
- [ ] Set TOKEN_EXPIRY_HOURS = 0.083
- [ ] Login
- [ ] Wait 5 minutes
- [ ] Check token is expired in console

### All Roles (10 minutes)
- [ ] Test Doctor login
- [ ] Test Pharma login
- [ ] Test Pharmacy login
- [ ] Test Hospital login
- [ ] Verify each has correct token

---

## 🚀 Files Modified

1. **`auth_config.py`** (NEW)
   - JWT token generation
   - Session management
   - Token validation

2. **`app.py`** (UPDATED)
   - Login endpoint returns JWT token
   - Added `/api/auth/refresh-token` endpoint
   - Session tracking

3. **`static/js/auth.js`** (UPDATED)
   - Token storage and retrieval
   - Session expiry checking
   - Token refresh logic

---

## 🎯 Key Features

| Feature | Duration | Status |
|---------|----------|--------|
| JWT Token | 24 hours | ✅ Implemented |
| Session Timeout | 30 minutes | ✅ Implemented |
| Token Refresh | On demand | ✅ Implemented |
| Session Refresh | On activity | ✅ Implemented |
| Multi-role support | All roles | ✅ Implemented |
| Logout cleanup | Immediate | ✅ Implemented |

---

## 🔗 Related Documentation

- Full guide: `docs/JWT_SESSION_TESTING_GUIDE.md`
- Auth config: `auth_config.py`
- Frontend auth: `static/js/auth.js`
- Backend auth: `app.py` (login endpoint)

---

## 💡 Tips

1. **For testing**: Use 1-minute session timeout and 5-minute token expiry
2. **For production**: Use 60-minute session timeout and 24-hour token expiry
3. **Always use HTTPS** in production (not HTTP)
4. **Change JWT_SECRET** before deploying to production
5. **Hash passwords** (currently plain text - security issue!)

---

**App Status**: ✅ Running on http://127.0.0.1:5000
**JWT Implementation**: ✅ Complete
**Session Management**: ✅ Complete
**Testing Guide**: ✅ Available in `docs/JWT_SESSION_TESTING_GUIDE.md`
