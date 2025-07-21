import React, { useState, useEffect } from 'react';
import { getJobs, cancelJob, getJobLogs } from '../api';

const RunTable = () => {
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [expandedLogs, setExpandedLogs] = useState(new Set());

    const loadJobs = async () => {
        try {
            const response = await getJobs();
            setJobs(response.data);
            setError(null);
        } catch (err) {
            setError('Failed to load jobs: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadJobs();
        const interval = setInterval(loadJobs, 2000); // Poll every 2 seconds
        return () => clearInterval(interval);
    }, []);

    const handleCancelJob = async (runId) => {
        try {
            await cancelJob(runId);
            await loadJobs(); // Refresh the list
        } catch (err) {
            setError('Failed to cancel job: ' + err.message);
        }
    };

    const toggleLogs = (runId) => {
        const newExpanded = new Set(expandedLogs);
        if (newExpanded.has(runId)) {
            newExpanded.delete(runId);
        } else {
            newExpanded.add(runId);
        }
        setExpandedLogs(newExpanded);
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'running':
                return 'text-blue-600 bg-blue-100';
            case 'completed':
                return 'text-green-600 bg-green-100';
            case 'failed':
                return 'text-red-600 bg-red-100';
            case 'cancelled':
                return 'text-gray-600 bg-gray-100';
            default:
                return 'text-yellow-600 bg-yellow-100';
        }
    };

    const formatDuration = (seconds) => {
        if (!seconds) return '-';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours}h ${minutes}m ${secs}s`;
    };

    if (loading) {
        return (
            <div className="flex justify-center items-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-2">Loading jobs...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
                <p className="text-red-600">{error}</p>
                <button
                    onClick={loadJobs}
                    className="mt-2 px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                >
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">Experiment Runs</h2>
                <button
                    onClick={loadJobs}
                    className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                >
                    Refresh
                </button>
            </div>

            {/* Jobs table */}
            {jobs.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                    No experiment runs found. Start an experiment to see it here!
                </div>
            ) : (
                <div className="bg-white shadow overflow-hidden rounded-md">
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Run ID
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Job Name
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Status
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Started
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Duration
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Experiments
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Actions
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {jobs.map((job) => (
                                    <React.Fragment key={job.run_id}>
                                        <tr className="hover:bg-gray-50">
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                                                {job.run_id.substring(0, 8)}...
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                                {job.job_name}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(job.status)}`}>
                                                    {job.status}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                                {new Date(job.started_at).toLocaleString()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                                {formatDuration(job.duration)}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                                {job.total_experiments}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                                <div className="flex space-x-2">
                                                    <button
                                                        onClick={() => toggleLogs(job.run_id)}
                                                        className="text-blue-600 hover:text-blue-800 text-sm"
                                                    >
                                                        {expandedLogs.has(job.run_id) ? 'Hide Logs' : 'Show Logs'}
                                                    </button>
                                                    {job.status === 'running' && (
                                                        <button
                                                            onClick={() => handleCancelJob(job.run_id)}
                                                            className="text-red-600 hover:text-red-800 text-sm"
                                                        >
                                                            Cancel
                                                        </button>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                        {expandedLogs.has(job.run_id) && (
                                            <tr>
                                                <td colSpan="7" className="px-6 py-4 bg-gray-50">
                                                    <LogPanel runId={job.run_id} />
                                                </td>
                                            </tr>
                                        )}
                                    </React.Fragment>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
};

// LogPanel component for displaying job logs
const LogPanel = ({ runId }) => {
    const [logs, setLogs] = useState('');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const response = await getJobLogs(runId);
                setLogs(response.data.logs);
            } catch (err) {
                setLogs('Failed to load logs: ' + err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchLogs();
        const interval = setInterval(fetchLogs, 2000); // Poll every 2 seconds
        return () => clearInterval(interval);
    }, [runId]);

    if (loading) {
        return (
            <div className="flex justify-center items-center h-32">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span className="ml-2">Loading logs...</span>
            </div>
        );
    }

    return (
        <div className="bg-black text-green-400 p-4 rounded-md font-mono text-sm max-h-96 overflow-y-auto">
            <pre className="whitespace-pre-wrap">{logs || 'No logs available'}</pre>
        </div>
    );
};

export default RunTable; 