import React, { useState, useEffect, useMemo } from 'react';
import { getResults, downloadResultsCSV } from '../api';

const ResultsGrid = () => {
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
    const [filters, setFilters] = useState({
        model: '',
        dataset: '',
        run_id: '',
        evaluation_metric: '',
        evaluation_tool: ''
    });

    const loadResults = async () => {
        try {
            const response = await getResults();
            setResults(response.data.results);
            setError(null);
        } catch (err) {
            setError('Failed to load results: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadResults();
    }, []);

    // Handle sorting
    const handleSort = (key) => {
        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    // Handle filtering
    const handleFilterChange = (key, value) => {
        setFilters(prev => ({ ...prev, [key]: value }));
    };

    // Apply filters and sorting
    const filteredAndSortedResults = useMemo(() => {
        let filtered = results.filter(result => {
            return (
                (filters.model === '' || result.model?.toLowerCase().includes(filters.model.toLowerCase())) &&
                (filters.dataset === '' || result.dataset?.toLowerCase().includes(filters.dataset.toLowerCase())) &&
                (filters.run_id === '' || result.run_id?.toLowerCase().includes(filters.run_id.toLowerCase())) &&
                (filters.evaluation_metric === '' || result.evaluation_metric === filters.evaluation_metric) &&
                (filters.evaluation_tool === '' || result.evaluation_tool === filters.evaluation_tool)
            );
        });

        if (sortConfig.key) {
            filtered.sort((a, b) => {
                const aValue = a[sortConfig.key];
                const bValue = b[sortConfig.key];

                if (aValue === null || aValue === undefined) return 1;
                if (bValue === null || bValue === undefined) return -1;

                if (typeof aValue === 'number' && typeof bValue === 'number') {
                    return sortConfig.direction === 'asc' ? aValue - bValue : bValue - aValue;
                }

                const aStr = String(aValue).toLowerCase();
                const bStr = String(bValue).toLowerCase();

                if (aStr < bStr) return sortConfig.direction === 'asc' ? -1 : 1;
                if (aStr > bStr) return sortConfig.direction === 'asc' ? 1 : -1;
                return 0;
            });
        }

        return filtered;
    }, [results, filters, sortConfig]);

    const downloadCSV = async () => {
        try {
            const response = await downloadResultsCSV();
            const blob = new Blob([response.data], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'experiment_results.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            alert('Failed to download CSV: ' + err.message);
        }
    };

    const clearFilters = () => {
        setFilters({ model: '', dataset: '', run_id: '', evaluation_metric: '', evaluation_tool: '' });
    };

    const getSortIcon = (key) => {
        if (sortConfig.key !== key) {
            return '⇅';
        }
        return sortConfig.direction === 'asc' ? '↑' : '↓';
    };

    const formatValue = (value) => {
        if (value === null || value === undefined) return '-';
        if (typeof value === 'number') {
            return value % 1 === 0 ? value.toString() : value.toFixed(4);
        }
        return value.toString();
    };

    if (loading) {
        return (
            <div className="flex justify-center items-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-2">Loading results...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
                <p className="text-red-600">{error}</p>
                <button
                    onClick={loadResults}
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
                <h2 className="text-xl font-semibold">Experiment Results</h2>
                <div className="flex space-x-2">
                    <button
                        onClick={loadResults}
                        className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                    >
                        Refresh
                    </button>
                    <button
                        onClick={downloadCSV}
                        className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700"
                    >
                        Download CSV
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div className="bg-gray-50 p-4 rounded-md">
                <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
                        <input
                            type="text"
                            value={filters.model}
                            onChange={(e) => handleFilterChange('model', e.target.value)}
                            className="w-full px-3 py-1 border border-gray-300 rounded text-sm"
                            placeholder="Filter by model..."
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Dataset</label>
                        <input
                            type="text"
                            value={filters.dataset}
                            onChange={(e) => handleFilterChange('dataset', e.target.value)}
                            className="w-full px-3 py-1 border border-gray-300 rounded text-sm"
                            placeholder="Filter by dataset..."
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Run ID</label>
                        <input
                            type="text"
                            value={filters.run_id}
                            onChange={(e) => handleFilterChange('run_id', e.target.value)}
                            className="w-full px-3 py-1 border border-gray-300 rounded text-sm"
                            placeholder="Filter by run ID..."
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Eval Metric</label>
                        <select
                            value={filters.evaluation_metric}
                            onChange={(e) => handleFilterChange('evaluation_metric', e.target.value)}
                            className="w-full px-3 py-1 border border-gray-300 rounded text-sm"
                        >
                            <option value="">All</option>
                            <option value="pass@k">pass@k</option>
                            <option value="maj@k">maj@k</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Eval Tool</label>
                        <select
                            value={filters.evaluation_tool}
                            onChange={(e) => handleFilterChange('evaluation_tool', e.target.value)}
                            className="w-full px-3 py-1 border border-gray-300 rounded text-sm"
                        >
                            <option value="">All</option>
                            <option value="rule-based">Rule-based</option>
                            <option value="llm">LLM</option>
                        </select>
                    </div>
                </div>
                <div className="mt-4">
                    <button
                        onClick={clearFilters}
                        className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700"
                    >
                        Clear Filters
                    </button>
                </div>
            </div>

            {/* Results count */}
            <div className="text-sm text-gray-600">
                Showing {filteredAndSortedResults.length} of {results.length} results
            </div>

            {/* Results table */}
            {results.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                    No experiment results found. Run some experiments to see results!
                </div>
            ) : (
                <div className="bg-white shadow overflow-hidden rounded-md">
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    {[
                                        { key: 'run_id', label: 'Run ID' },
                                        { key: 'model', label: 'Model' },
                                        { key: 'dataset', label: 'Dataset' },
                                        { key: 'split', label: 'Split' },
                                        { key: 'temp', label: 'Temperature' },
                                        { key: 'top_p', label: 'Top P' },
                                        { key: 'top_k', label: 'Top K' },
                                        { key: 'seed', label: 'Seed' },
                                        { key: 'accuracy', label: 'Accuracy' },
                                        { key: 'loss', label: 'Loss' },
                                        { key: 'runtime', label: 'Runtime (s)' },
                                        { key: 'evaluation_metric', label: 'Eval Metric' },
                                        { key: 'at_k_value', label: '@k' },
                                        { key: 'evaluation_tool', label: 'Eval Tool' },
                                        { key: 'judge_model', label: 'Judge Model' },
                                        { key: 'timestamp', label: 'Timestamp' }
                                    ].map(({ key, label }) => (
                                        <th
                                            key={key}
                                            onClick={() => handleSort(key)}
                                            className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                                        >
                                            <div className="flex items-center space-x-1">
                                                <span>{label}</span>
                                                <span className="text-gray-400">{getSortIcon(key)}</span>
                                            </div>
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {filteredAndSortedResults.map((result, index) => (
                                    <tr key={index} className="hover:bg-gray-50">
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                                            {result.run_id ? result.run_id.substring(0, 8) + '...' : '-'}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.model)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.dataset)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.split)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.temp)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.top_p)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.top_k)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.seed)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {result.accuracy !== null && result.accuracy !== undefined ? (
                                                <span className={`font-medium ${result.accuracy > 0.8 ? 'text-green-600' : result.accuracy > 0.6 ? 'text-yellow-600' : 'text-red-600'}`}>
                                                    {formatValue(result.accuracy)}
                                                </span>
                                            ) : '-'}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.loss)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.runtime)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.evaluation_metric)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.at_k_value)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.evaluation_tool)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {formatValue(result.judge_model)}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                            {result.timestamp ? new Date(result.timestamp).toLocaleString() : '-'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ResultsGrid; 