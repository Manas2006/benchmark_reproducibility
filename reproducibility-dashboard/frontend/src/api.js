import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// API functions
export const runExperiment = async (experimentData) => {
    const response = await api.post('/run', experimentData);
    return response;
};

export const getJobs = async () => {
    const response = await api.get('/jobs');
    return response;
};

export const getJobLogs = async (runId) => {
    const response = await api.get(`/jobs/${runId}/logs`);
    return response;
};

export const cancelJob = async (runId) => {
    const response = await api.delete(`/jobs/${runId}`);
    return response;
};

export const getResults = async () => {
    const response = await api.get('/results');
    return response;
};

export const downloadResultsCSV = async () => {
    const response = await api.get('/results/download', {
        responseType: 'blob',
    });
    return response;
};

export const runUnitTest = async (unitTestData) => {
    const response = await api.post('/unit_test', unitTestData);
    return response;
};

export default api; 