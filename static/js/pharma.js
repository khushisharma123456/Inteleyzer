/**
 * Pharma Intelligence Logic (Connected to Flask API)
 */

export const PharmaApp = {
    getAllData: async () => {
        // Reuse patient endpoint which returns all for Pharma role
        const res = await fetch('/api/patients');
        return await res.json();
    },

    getAggregateStats: async () => {
        const res = await fetch('/api/stats');
        return await res.json();
    }
};
