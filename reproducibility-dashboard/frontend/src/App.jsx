import React, { useState } from 'react';
import RawSbatchInput from './components/RawSbatchInput';
import SettingsInput from './components/SettingsInput';
import EnvironmentInput from './components/EnvironmentInput';
import GenerationConfig from './components/GenerationConfig';
import EvaluationConfig from './components/EvaluationConfig';
import RunTable from './components/RunTable';
import ResultsGrid from './components/ResultsGrid';
import UnitTestButton from './components/UnitTestButton';
import { runExperiment } from './api';

function App() {
    const [activeTab, setActiveTab] = useState('configure');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [message, setMessage] = useState('');

    // Form state
    const [formData, setFormData] = useState({
        raw_sbatch_directives: `#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1`,
        local_dir: '/scratch/user/experiments',
        output_dir: './experiment_outputs',
        environment_settings: 'cd /scratch/user/experiments\nconda activate eval_repro',

        // Generation config
        models: ['gpt-3.5-turbo', 'claude-3-haiku'],
        datasets: ['gsm8k|test', 'math|validation'],
        top_ps: [0.9, 0.95],
        top_ks: [1, 5],
        temps: [0.0, 0.3, 0.7],
        max_lengths: [2048],
        max_new_tokens: [512],
        seeds: [42, 123, 456],
        prompt: 'Answer the following question: {question}',

        // Evaluation config
        evaluation_metric: 'pass@k',
        at_k_value: 1,
        evaluation_prompt: 'Evaluate if the answer is correct.',
        evaluation_tool: 'rule-based',
        extraction_method: 'predefined',
        predefined_extractor: 'boxed_answer',
        judge_model_type: 'api',
        judge_model: '',
        judge_api_key: '',
        local_llm_path: '',
        custom_extractor_code: '',

        // Optional existing results
        existing_result_path: ''
    });

    const [errors, setErrors] = useState({});

    const handleInputChange = (field, value) => {
        setFormData(prev => ({ ...prev, [field]: value }));
        // Clear error when user starts typing
        if (errors[field]) {
            setErrors(prev => ({ ...prev, [field]: null }));
        }
    };

    const validateForm = () => {
        const newErrors = {};

        if (!formData.raw_sbatch_directives.trim()) {
            newErrors.raw_sbatch_directives = 'SBATCH directives are required';
        }
        if (!formData.local_dir.trim()) {
            newErrors.local_dir = 'Local directory is required';
        }
        if (!formData.output_dir.trim()) {
            newErrors.output_dir = 'Output directory is required';
        }
        if (!formData.environment_settings.trim()) {
            newErrors.environment_settings = 'Environment settings are required';
        }

        // Filter out empty lines for validation
        const filteredModels = formData.models.filter(line => String(line).trim() !== '');
        const filteredDatasets = formData.datasets.filter(line => String(line).trim() !== '');
        const filteredTopPs = formData.top_ps.filter(line => String(line).trim() !== '');
        const filteredTopKs = formData.top_ks.filter(line => String(line).trim() !== '');
        const filteredTemps = formData.temps.filter(line => String(line).trim() !== '');
        const filteredSeeds = formData.seeds.filter(line => String(line).trim() !== '');

        if (!filteredModels.length) {
            newErrors.models = 'At least one model is required';
        }
        if (!filteredDatasets.length) {
            newErrors.datasets = 'At least one dataset is required';
        }
        if (!filteredTopPs.length) {
            newErrors.top_ps = 'At least one top_p value is required';
        }
        if (!filteredTopKs.length) {
            newErrors.top_ks = 'At least one top_k value is required';
        }
        if (!filteredTemps.length) {
            newErrors.temps = 'At least one temperature value is required';
        }
        if (!filteredSeeds.length) {
            newErrors.seeds = 'At least one seed is required';
        }
        if (!formData.at_k_value || formData.at_k_value < 1) {
            newErrors.at_k_value = '@k value must be at least 1';
        }
        if (formData.evaluation_tool === 'llm') {
            if (formData.judge_model_type === 'api' && !formData.judge_model.trim()) {
                newErrors.judge_model = 'Judge model is required when using API request';
            }
            if (formData.judge_model_type === 'local' && !formData.local_llm_path.trim()) {
                newErrors.local_llm_path = 'Local LLM path is required when using local model';
            }
        }
        if (formData.evaluation_tool === 'rule-based' && formData.extraction_method === 'custom' && !formData.custom_extractor_code.trim()) {
            newErrors.custom_extractor_code = 'Custom extractor code is required when using custom extraction method';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!validateForm()) {
            setMessage('Please fix the errors above');
            return;
        }

        setIsSubmitting(true);
        setMessage('');

        try {
            const response = await runExperiment(formData);
            setMessage(`✅ ${response.data.message}`);

            // Switch to runs tab to see the progress
            setTimeout(() => {
                setActiveTab('runs');
            }, 2000);

        } catch (error) {
            setMessage(`❌ Error: ${error.response?.data?.detail || error.message}`);
        } finally {
            setIsSubmitting(false);
        }
    };

    const calculateTotalExperiments = () => {
        if (formData.existing_result_path) {
            return 1; // Only evaluation
        }
        // Filter out empty lines for calculation
        const filteredModels = formData.models.filter(line => String(line).trim() !== '');
        const filteredDatasets = formData.datasets.filter(line => String(line).trim() !== '');
        const filteredTemps = formData.temps.filter(line => String(line).trim() !== '');
        const filteredTopPs = formData.top_ps.filter(line => String(line).trim() !== '');
        const filteredTopKs = formData.top_ks.filter(line => String(line).trim() !== '');
        const filteredSeeds = formData.seeds.filter(line => String(line).trim() !== '');

        return filteredModels.length *
            filteredDatasets.length *
            filteredTemps.length *
            filteredTopPs.length *
            filteredTopKs.length *
            filteredSeeds.length;
    };

    const tabs = [
        { id: 'configure', label: 'Configure & Run', icon: '⚙️' },
        { id: 'runs', label: 'Runs & Logs', icon: '🏃' },
        { id: 'results', label: 'Results & Export', icon: '📊' }
    ];

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white shadow-sm border-b">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center py-4">
                        <h1 className="text-2xl font-bold text-gray-900">
                            🧪 Reproducibility Dashboard
                        </h1>
                    </div>
                </div>
            </header>

            {/* Tab Navigation */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="border-b border-gray-200">
                    <nav className="-mb-px flex space-x-8">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === tab.id
                                    ? 'border-blue-500 text-blue-600'
                                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    }`}
                            >
                                <span className="mr-2">{tab.icon}</span>
                                {tab.label}
                            </button>
                        ))}
                    </nav>
                </div>
            </div>

            {/* Tab Content */}
            <main className="w-full">
                {activeTab === 'configure' && (
                    <div className="w-full">
                        <div className="bg-white shadow rounded-lg p-8">
                            <div className="mb-8">
                                <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                                    Configure Experiment
                                </h2>
                                <p className="text-gray-600">
                                    Set up your hyperparameter sweep experiment. This will generate and run {calculateTotalExperiments()} individual experiments.
                                </p>
                            </div>

                            <form onSubmit={handleSubmit} className="space-y-8">
                                {/* SBATCH Configuration */}
                                <div>
                                    <h3 className="text-lg font-medium text-gray-900 mb-4">SBATCH Configuration</h3>
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                        <div className="lg:col-span-2">
                                            <RawSbatchInput
                                                value={formData.raw_sbatch_directives}
                                                onChange={(e) => handleInputChange('raw_sbatch_directives', e.target.value)}
                                                error={errors.raw_sbatch_directives}
                                            />
                                        </div>
                                        <div className="lg:col-span-2">
                                            <EnvironmentInput
                                                label="Environment Settings"
                                                value={formData.environment_settings}
                                                onChange={(e) => handleInputChange('environment_settings', e.target.value)}
                                                placeholder="cd /scratch/user/experiments&#10;conda activate eval_repro&#10;export CUDA_VISIBLE_DEVICES=0"
                                                error={errors.environment_settings}
                                                required
                                                helperText="Commands to run before executing experiments (e.g., cd, conda activate, export variables)"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Directory Configuration */}
                                <div>
                                    <h3 className="text-lg font-medium text-gray-900 mb-4">Directory Configuration</h3>
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                        <SettingsInput
                                            label="Local Directory"
                                            value={formData.local_dir}
                                            onChange={(e) => handleInputChange('local_dir', e.target.value)}
                                            placeholder="/scratch/user/experiments"
                                            error={errors.local_dir}
                                            required
                                            helperText="Directory to cd into before running experiments"
                                        />
                                        <SettingsInput
                                            label="Output Directory"
                                            value={formData.output_dir}
                                            onChange={(e) => handleInputChange('output_dir', e.target.value)}
                                            placeholder="./experiment_outputs"
                                            error={errors.output_dir}
                                            required
                                            helperText="Directory to store experiment outputs"
                                        />
                                    </div>
                                </div>

                                {/* Generation Configuration */}
                                <GenerationConfig
                                    formData={formData}
                                    handleInputChange={handleInputChange}
                                    errors={errors}
                                />

                                {/* Evaluation Configuration */}
                                <EvaluationConfig
                                    formData={formData}
                                    handleInputChange={handleInputChange}
                                    errors={errors}
                                />

                                {/* Existing Results (Optional) */}
                                <div>
                                    <h3 className="text-lg font-medium text-gray-900 mb-4">Existing Results (Optional)</h3>
                                    <SettingsInput
                                        label="Existing Results Path"
                                        value={formData.existing_result_path}
                                        onChange={(e) => handleInputChange('existing_result_path', e.target.value)}
                                        placeholder="/path/to/existing/results.json"
                                        helperText="Path to existing results to evaluate (skip generation)"
                                    />
                                </div>

                                {/* Summary */}
                                <div className="bg-blue-50 rounded-md p-4">
                                    <div className="flex">
                                        <div className="ml-3">
                                            <h3 className="text-sm font-medium text-blue-800">
                                                Experiment Summary
                                            </h3>
                                            <div className="mt-2 text-sm text-blue-700">
                                                <p>Total experiments to run: <strong>{calculateTotalExperiments()}</strong></p>
                                                <p>This will test {formData.models.length} models on {formData.datasets.length} datasets
                                                    with {formData.temps.length} temperatures, {formData.top_ps.length} top_p values,
                                                    {formData.top_ks.length} top_k values, and {formData.seeds.length} seeds.</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Action Buttons */}
                                <div className="flex justify-between items-center">
                                    <UnitTestButton formData={formData} />
                                    <button
                                        type="submit"
                                        disabled={isSubmitting}
                                        className="px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {isSubmitting ? 'Starting Experiment...' : 'Generate & Run Experiment'}
                                    </button>
                                </div>

                                {message && (
                                    <div className={`mt-4 p-4 rounded-md ${message.startsWith('✅') ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
                                        }`}>
                                        {message}
                                    </div>
                                )}
                            </form>
                        </div>
                    </div>
                )}

                {activeTab === 'runs' && <RunTable />}
                {activeTab === 'results' && <ResultsGrid />}
            </main>
        </div>
    );
}

export default App; 