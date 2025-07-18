import React from 'react';
import ListInput from './ListInput';

const GenerationConfig = ({ formData, handleInputChange, errors }) => {
    return (
        <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Generation Configuration</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ListInput
                    label="Models"
                    value={formData.models}
                    onChange={(value) => handleInputChange('models', value)}
                    placeholder="gpt-3.5-turbo&#10;claude-3-haiku&#10;llama-2-7b"
                    error={errors.models}
                    helperText="List of models to test (one per line)"
                    required
                />
                <ListInput
                    label="Datasets"
                    value={formData.datasets}
                    onChange={(value) => handleInputChange('datasets', value)}
                    placeholder="gsm8k|test&#10;math|validation&#10;mmlu|test"
                    error={errors.datasets}
                    helperText="List of datasets to evaluate on (format: dataset|split)"
                    required
                />
                <ListInput
                    label="Top P Values"
                    value={formData.top_ps}
                    onChange={(value) => handleInputChange('top_ps', value)}
                    type="number"
                    placeholder="0.9&#10;0.95&#10;1.0"
                    error={errors.top_ps}
                    helperText="List of top_p values (0-1)"
                    required
                />
                <ListInput
                    label="Top K Values"
                    value={formData.top_ks}
                    onChange={(value) => handleInputChange('top_ks', value)}
                    type="integer"
                    placeholder="1&#10;5&#10;10"
                    error={errors.top_ks}
                    helperText="List of top_k values"
                    required
                />
                <ListInput
                    label="Temperature Values"
                    value={formData.temps}
                    onChange={(value) => handleInputChange('temps', value)}
                    type="number"
                    placeholder="0.0&#10;0.3&#10;0.7&#10;1.0"
                    error={errors.temps}
                    helperText="List of temperature values"
                    required
                />
                <ListInput
                    label="Max Model Lengths"
                    value={formData.max_lengths}
                    onChange={(value) => handleInputChange('max_lengths', value)}
                    type="integer"
                    placeholder="2048&#10;4096"
                    helperText="List of max model lengths (optional)"
                />
                <ListInput
                    label="Max New Tokens"
                    value={formData.max_new_tokens}
                    onChange={(value) => handleInputChange('max_new_tokens', value)}
                    type="integer"
                    placeholder="512&#10;1024"
                    helperText="List of max new tokens (optional)"
                />
                <ListInput
                    label="Random Seeds"
                    value={formData.seeds}
                    onChange={(value) => handleInputChange('seeds', value)}
                    type="integer"
                    placeholder="42&#10;123&#10;456"
                    error={errors.seeds}
                    helperText="List of random seeds for reproducibility"
                    required
                />
                <div className="lg:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Generation Prompt
                    </label>
                    <textarea
                        value={formData.prompt}
                        onChange={(e) => handleInputChange('prompt', e.target.value)}
                        rows={6}
                        className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${errors.prompt ? 'border-red-300' : 'border-gray-300'
                            }`}
                        placeholder="Enter your generation prompt here. This will be used to guide the model's responses."
                    />
                    {errors.prompt && (
                        <p className="mt-1 text-sm text-red-600">{errors.prompt}</p>
                    )}
                    <p className="mt-1 text-xs text-gray-500">
                        Prompt that will be used for generation. Use placeholders like {'{question}'} for dynamic content.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default GenerationConfig; 