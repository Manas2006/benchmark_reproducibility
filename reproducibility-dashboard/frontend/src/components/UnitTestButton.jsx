import React, { useState } from 'react';
import { runUnitTest } from '../api';

const UnitTestButton = ({ formData }) => {
    const [isRunning, setIsRunning] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleUnitTest = async () => {
        setIsRunning(true);
        setResult(null);
        setError(null);

        try {
            // Prepare unit test data
            const unitTestData = {
                model: formData.models[0] || 'test-model',
                dataset: formData.datasets[0] || 'test_dataset',
                split: formData.datasets[0]?.includes('|') ? formData.datasets[0].split('|')[1] : 'test',
                temperature: formData.temps[0] || 0.0,
                top_p: formData.top_ps[0] || 1.0,
                top_k: formData.top_ks[0] || 1,
                seed: formData.seeds[0] || 42,
                max_length: formData.max_lengths[0] || 2048,
                max_new_tokens: formData.max_new_tokens[0] || 512,
                prompt: formData.prompt || 'Answer the following question:',
                local_dir: formData.local_dir || '/testing'
            };

            const response = await runUnitTest(unitTestData);
            setResult(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || err.message);
        } finally {
            setIsRunning(false);
        }
    };

    return (
        <div className="space-y-4">
            <button
                onClick={handleUnitTest}
                disabled={isRunning}
                className="px-4 py-2 bg-purple-600 text-white font-medium rounded-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {isRunning ? 'Running Unit Test...' : 'Run Unit Test'}
            </button>

            {result && (
                <div className={`p-4 rounded-md ${result.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
                    <h4 className={`font-medium ${result.success ? 'text-green-800' : 'text-red-800'}`}>
                        Unit Test {result.success ? 'Passed' : 'Failed'}
                    </h4>
                    {result.output && (
                        <div className="mt-2">
                            <h5 className="text-sm font-medium text-gray-700">Output:</h5>
                            <pre className="mt-1 text-sm bg-gray-100 p-2 rounded overflow-x-auto">
                                {result.output}
                            </pre>
                        </div>
                    )}
                    {result.error && (
                        <div className="mt-2">
                            <h5 className="text-sm font-medium text-red-700">Error:</h5>
                            <pre className="mt-1 text-sm text-red-600 bg-red-50 p-2 rounded overflow-x-auto">
                                {result.error}
                            </pre>
                        </div>
                    )}
                </div>
            )}

            {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-md">
                    <h4 className="font-medium text-red-800">Unit Test Error</h4>
                    <p className="mt-1 text-sm text-red-600">{error}</p>
                </div>
            )}

            <p className="text-xs text-gray-500">
                Unit tests run a single experiment with the first values from your configuration to validate your setup before running the full sweep.
            </p>
        </div>
    );
};

export default UnitTestButton; 