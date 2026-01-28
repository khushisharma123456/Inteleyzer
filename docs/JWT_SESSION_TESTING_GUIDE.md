# JWT Token & Session Expiry Testing Guide

## Overview

The authentication system now includes:
- **JWT Tokens**: 24-hour expiry
- **Session Timeout**: 30 minutes of inactivity
- **Token Refresh**: Automatic token refresh before expiry
- **Session Monitoring**: Real-time session expiry checks

---

## Configuration

### Current Settings

```python
# auth_config.py
SESSION_TIMEOUT_MINUTES = 30      # Session expires after 30 minutes
TOKEN_EXPIRY_HOURS = 24           # JWT token expires after 24 hours
```

### How to Change Settings

Edit `auth_config.py`:

```python
# For testing (shorter timeouts)
SESSION_TIMEOUT_MINUTES = 1       # 1 minute session timeout
TOKEN_EXPIRY_HOURS = 0.083        # 5 minutes token expiry (0.083 hours)

# For production (longer timeouts)
SESSION_TIMEOUT_MINUTES = 60      # 1 hour session timeout
TOKEN_EXPIRY_HOURS = 24           # 24 hours token expiry
```

---

## Testing Scenarios

### Test 1: Login and Verify JWT Token

**Duration**: 2 minutes

**Steps**:

1. **Start the app**:
   ```bash
   python app.py
   ```

2. **Open browser and navigate to**:
   ```
   http://127.0.0.1:5000/login
   ```

3. **Login with test credentials**:
   - Email: `doctor@hospital.com`
   - Password: `password123`
   - Role: Doctor

4. **Verify token in browser console**:
   ```javascript
   // Open DevTools (F12) → Console
   localStorage.getItem('jwt_token')
   // Should return a long JWT token string
   
   localStorage.getItem('token_expiry')
   // Should return a timestamp
   ```

5. **Decode JWT token** (optional):
   - Go to https://jwt.io
   - Paste the token in the "Encoded" section
   - Verify payload contains: `user_id`, `email`, `role`, `name`, `exp`

**Expected Result**: ✅ Token is generated and stored in localStorage

---

### Test 2: Session Timeout (30 minutes)

**Duration**: 31 minutes (or use shorter timeout for testing)

**For Quick Testing (1 minute timeout)**:

1. **Edit `auth_config.py`**:
   ```python
   SESSION_TIMEOUT_MINUTES = 1  # 1 minute instead of 30
   ```

2. **Restart app**:
   ```bash
   # Stop current app (Ctrl+C)
   python app.py
   ```

3. **Login again**:
   - Navigate to http://127.0.0.1:5000/login
   - Login with credentials

4. **Wait 1 minute** (or 30 minutes for production testing)

5. **Try to access a protected page**:
   - Click on "Patients" or any dashboard link
   - You should be redirected to login page

6. **Check browser console**:
   ```javascript
   localStorage.getItem('session_start')
   // Should be cleared after timeout
   ```

**Expected Result**: ✅ Session expires and user is logged out

---

### Test 3: Token Expiry (24 hours)

**Duration**: 24 hours (or use shorter timeout for testing)

**For Quick Testing (5 minutes token expiry)**:

1. **Edit `auth_config.py`**:
   ```python
   TOKEN_EXPIRY_HOURS = 0.083  # 5 minutes instead of 24 hours
   ```

2. **Restart app and login**

3. **Wait 5 minutes**

4. **Try to make an API call**:
   ```javascript
   // In browser console
   const token = localStorage.getItem('jwt_token');
   const expiry = localStorage.getItem('token_expiry');
   const now = new Date().getTime();
   console.log('Token expired:', now > parseInt(expiry));
   ```

5. **Expected**: Token should be expired

**Expected Result**: ✅ Token expires after configured time

---

### Test 4: Token Refresh

**Duration**: 5 minutes

**Steps**:

1. **Set short token expiry** in `auth_config.py`:
   ```python
   TOKEN_EXPIRY_HOURS = 0.083  # 5 minutes
   ```

2. **Login and note the token**:
   ```javascript
   const token1 = localStorage.getItem('jwt_token');
   console.log('Initial token:', token1);
   ```

3. **Call refresh endpoint** (before token expires):
   ```javascript
   const response = await fetch('/api/auth/refresh-token', {
       method: 'POST',
       headers: {
           'Content-Type': 'application/json',
           'Authorization': `Bearer ${localStorage.getItem('jwt_token')}`
       }
   });
   const data = await response.json();
   console.log('New token:', data.token);
   ```

4. **Verify new token is different**:
   ```javascript
   const token2 = localStorage.getItem('jwt_token');
   console.log('Token refreshed:', token1 !== token2);
   ```

**Expected Result**: ✅ New token is generated and stored

---

### Test 5: Session Activity Refresh

**Duration**: 5 minutes

**Steps**:

1. **Set 1-minute session timeout**:
   ```python
   SESSION_TIMEOUT_MINUTES = 1
   ```

2. **Login**

3. **Wait 30 seconds** (half the timeout)

4. **Click on a page or make an API call**:
   - This should refresh the session
   ```javascript
   // In console
   localStorage.getItem('session_start')
   // Should show recent timestamp
   ```

5. **Wait another 30 seconds** (total 60 seconds)

6. **Without any activity, wait 30 more seconds** (total 90 seconds)

7. **Try to access a page**:
   - Should be logged out

**Expected Result**: ✅ Session is refreshed on activity, expires on inactivity

---

### Test 6: Multiple User Roles

**Duration**: 10 minutes

**Steps**:

1. **Test each role**:
   - Doctor: `doctor@hospital.com`
   - Pharma: `pharma@company.com`
   - Pharmacy: `pharmacy@store.com`
   - Hospital: `hospital@health.com`

2. **For each role**:
   - Login
   - Verify token in console
   - Check role in localStorage: `localStorage.getItem('user_role')`
   - Verify correct dashboard loads

3. **Verify token contains correct role**:
   ```javascript
   // Decode token at jwt.io
   // Check "role" field matches user role
   ```

**Expected Result**: ✅ Each role gets correct token with role information

---

### Test 7: Logout and Token Cleanup

**Duration**: 2 minutes

**Steps**:

1. **Login**

2. **Verify token exists**:
   ```javascript
   console.log('Token before logout:', localStorage.getItem('jwt_token'));
   ```

3. **Click Logout button** (in sidebar)

4. **Verify token is cleared**:
   ```javascript
   console.log('Token after logout:', localStorage.getItem('jwt_token'));
   // Should be null
   ```

5. **Verify redirected to login page**

**Expected Result**: ✅ Token and session are cleared on logout

---

## Testing Checklist

### Quick Test (5 minutes)
- [ ] Login successfully
- [ ] JWT token is generated
- [ ] Token is stored in localStorage
- [ ] Logout clears token
- [ ] Redirected to login after logout

### Medium Test (35 minutes)
- [ ] All quick tests pass
- [ ] Session timeout works (set to 1 minute)
- [ ] User is logged out after inactivity
- [ ] Session is refreshed on activity

### Full Test (24+ hours)
- [ ] All medium tests pass
- [ ] Token expiry works (set to 5 minutes)
- [ ] Token refresh works
- [ ] All user roles work correctly

---

## Browser Console Commands

### Check Token Status
```javascript
// Check if token exists
localStorage.getItem('jwt_token') ? 'Token exists' : 'No token'

// Check token expiry
const expiry = parseInt(localStorage.getItem('token_expiry'));
const now = new Date().getTime();
const minutesLeft = Math.round((expiry - now) / 60000);
console.log(`Token expires in ${minutesLeft} minutes`);

// Check session status
const sessionStart = parseInt(localStorage.getItem('session_start'));
const sessionTimeout = parseInt(localStorage.getItem('session_timeout'));
const sessionAge = now - sessionStart;
const minutesUntilExpiry = Math.round((sessionTimeout - sessionAge) / 60000);
console.log(`Session expires in ${minutesUntilExpiry} minutes`);
```

### Manually Expire Token (for testing)
```javascript
// Set token expiry to now (expires immediately)
localStorage.setItem('token_expiry', new Date().getTime());

// Try to access a page - should redirect to login
window.location.href = '/doctor/dashboard';
```

### Manually Expire Session (for testing)
```javascript
// Set session start to 31 minutes ago (with 30-minute timeout)
const thirtyOneMinutesAgo = new Date().getTime() - (31 * 60 * 1000);
localStorage.setItem('session_start', thirtyOneMinutesAgo);

// Try to access a page - should redirect to login
window.location.href = '/doctor/dashboard';
```

---

## API Testing with cURL

### Login and Get Token
```bash
curl -X POST http://127.0.0.1:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "doctor@hospital.com",
    "password": "password123"
  }'

# Response:
# {
#   "success": true,
#   "user": {...},
#   "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
#   "token_expiry": 24,
#   "session_timeout": 30
# }
```

### Use Token in API Call
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."

curl -X GET http://127.0.0.1:5000/api/patients \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"
```

### Refresh Token
```bash
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."

curl -X POST http://127.0.0.1:5000/api/auth/refresh-token \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"
```

---

## Troubleshooting

### Issue: Token not being generated
**Solution**: 
- Check `auth_config.py` is imported in `app.py`
- Verify JWT_SECRET is set
- Check Flask app is running

### Issue: Session expires too quickly
**Solution**:
- Increase `SESSION_TIMEOUT_MINUTES` in `auth_config.py`
- Default is 30 minutes (production)
- For testing, use 1-5 minutes

### Issue: Token not being sent in API calls
**Solution**:
- Use `Auth.getAuthHeader()` in JavaScript
- Include `Authorization: Bearer <token>` header
- Check token is not expired

### Issue: "Token expired or invalid" error
**Solution**:
- Token may have expired - call `/api/auth/refresh-token`
- Or login again to get new token
- Check token expiry time in localStorage

---

## Production Recommendations

### Security Best Practices

1. **Use HTTPS only** (not HTTP)
   ```python
   app.config['SESSION_COOKIE_SECURE'] = True  # Only send over HTTPS
   ```

2. **Increase token expiry** for production
   ```python
   TOKEN_EXPIRY_HOURS = 24  # 24 hours
   SESSION_TIMEOUT_MINUTES = 60  # 1 hour
   ```

3. **Use strong JWT secret**
   ```python
   JWT_SECRET = os.environ.get('JWT_SECRET')  # From environment variable
   ```

4. **Enable CSRF protection**
   ```python
   from flask_wtf.csrf import CSRFProtect
   csrf = CSRFProtect(app)
   ```

5. **Hash passwords** (currently plain text - FIX THIS!)
   ```python
   from werkzeug.security import generate_password_hash, check_password_hash
   ```

---

## Summary

| Feature | Duration | How to Test |
|---------|----------|------------|
| JWT Token Generation | Immediate | Login and check localStorage |
| Token Expiry | 24 hours (default) | Set to 5 min, wait 5 min |
| Session Timeout | 30 minutes (default) | Set to 1 min, wait 1 min |
| Token Refresh | On demand | Call `/api/auth/refresh-token` |
| Session Refresh | On activity | Click page, check session_start |
| Logout | Immediate | Click logout, verify token cleared |

---

**Last Updated**: January 28, 2024
**Version**: 1.0
