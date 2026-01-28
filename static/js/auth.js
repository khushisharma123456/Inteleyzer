/**
 * Auth Logic for Inteleyzer (Connected to Flask API)
 * Includes JWT Token Management and Session Expiry
 */

export const Auth = {
    // Token management
    setToken: (token, expiryHours) => {
        localStorage.setItem('jwt_token', token);
        const expiryTime = new Date().getTime() + (expiryHours * 60 * 60 * 1000);
        localStorage.setItem('token_expiry', expiryTime);
    },

    getToken: () => {
        return localStorage.getItem('jwt_token');
    },

    isTokenExpired: () => {
        const expiry = localStorage.getItem('token_expiry');
        if (!expiry) return true;
        return new Date().getTime() > parseInt(expiry);
    },

    // Session management
    setSessionTimeout: (timeoutMinutes) => {
        const sessionStart = new Date().getTime();
        localStorage.setItem('session_start', sessionStart);
        localStorage.setItem('session_timeout', timeoutMinutes * 60 * 1000);
    },

    isSessionExpired: () => {
        const sessionStart = localStorage.getItem('session_start');
        const sessionTimeout = localStorage.getItem('session_timeout');
        
        if (!sessionStart || !sessionTimeout) return true;
        
        const elapsed = new Date().getTime() - parseInt(sessionStart);
        return elapsed > parseInt(sessionTimeout);
    },

    refreshSession: () => {
        const sessionTimeout = localStorage.getItem('session_timeout');
        if (sessionTimeout) {
            const sessionStart = new Date().getTime();
            localStorage.setItem('session_start', sessionStart);
        }
    },

    register: async (name, email, password, role) => {
        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email, password, role })
            });
            const data = await response.json();
            if (!response.ok) {
                return { success: false, message: data.message || 'Registration failed' };
            }
            return data;
        } catch (error) {
            console.error('Registration error:', error);
            return { success: false, message: 'Network error: ' + error.message };
        }
    },

    login: async (email, password) => {
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const result = await response.json();
            if (result.success) {
                // Store JWT token
                Auth.setToken(result.token, result.token_expiry);
                
                // Set session timeout
                Auth.setSessionTimeout(result.session_timeout);
                
                // Store basic info for UI update
                localStorage.setItem('user_name', result.user.name);
                localStorage.setItem('user_role', result.user.role);
                localStorage.setItem('user_email', result.user.email);
                localStorage.setItem('user_id', result.user.id);
                
                // Start session expiry check
                Auth.startSessionExpiryCheck();
            }
            return result;
        } catch (error) {
            console.error('Login error:', error);
            return { success: false, message: 'Network error: ' + error.message };
        }
    },

    refreshToken: async () => {
        try {
            const response = await fetch('/api/auth/refresh-token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${Auth.getToken()}`
                }
            });

            const result = await response.json();
            if (result.success) {
                Auth.setToken(result.token, result.token_expiry);
                Auth.setSessionTimeout(result.session_timeout);
                return true;
            }
            return false;
        } catch (error) {
            console.error('Token refresh error:', error);
            return false;
        }
    },

    startSessionExpiryCheck: () => {
        // Check session expiry every minute
        setInterval(() => {
            if (Auth.isSessionExpired()) {
                Auth.logout();
            }
        }, 60000); // Check every 60 seconds
    },

    logout: async () => {
        try {
            await fetch('/api/auth/logout', { method: 'POST' });
        } catch (error) {
            console.error('Logout error:', error);
        }
        
        localStorage.clear();
        window.location.href = '/login';
    },

    requireAuth: () => {
        const role = localStorage.getItem('user_role');
        const name = localStorage.getItem('user_name');

        if (!role || Auth.isSessionExpired()) {
            Auth.logout();
            return null;
        }
        
        // Refresh session on activity
        Auth.refreshSession();
        
        return { name, role };
    },

    // Get authorization header for API calls
    getAuthHeader: () => {
        const token = Auth.getToken();
        if (token && !Auth.isTokenExpired()) {
            return {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            };
        }
        return { 'Content-Type': 'application/json' };
    }
};
