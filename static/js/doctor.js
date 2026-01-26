/**
 * Doctor Dashboard Logic (Connected to Flask API)
 */

export const DoctorApp = {
    getPatients: async () => {
        const res = await fetch('/api/patients');
        return await res.json();
    },

    createPatient: async (data) => {
        const res = await fetch('/api/patients', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return await res.json();
    },

    renderRiskBadge: (level) => {
        const type = level.toLowerCase();
        return `<span class="badge badge-${type}">${level} Risk</span>`;
    }
};
