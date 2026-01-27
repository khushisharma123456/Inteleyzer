/**
 * Auth Logic for Inteleyzer (Connected to Flask API)
 */

export const Auth = {
    register: async (name, email, password, role) => {
        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ full_name: name, email, password, role })
            });
            return await response.json();
        } catch (error) {
            return { success: false, message: 'Network error' };
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
                // Store basic info for UI update (actual session is HttpOnly cookie managed by Flask)
                localStorage.setItem('user_name', result.user.full_name);
                localStorage.setItem('user_role', result.user.role);
            }
            return result;
        } catch (error) {
            return { success: false, message: 'Network error' };
        }
    },

    logout: async () => {
        await fetch('/api/auth/logout', { method: 'POST' });
        localStorage.clear();
        window.location.href = '/login';
    },

    requireAuth: () => {
        // In real app, we check session endpoint. For UI speed, we check localStorage hint.
        // Backend routes protect the HTML anyway via redirect.
        const role = localStorage.getItem('user_role');
        const name = localStorage.getItem('user_name');

        if (!role) {
            window.location.href = '/login';
            return null;
        }
        return { name, role };
    }
};
