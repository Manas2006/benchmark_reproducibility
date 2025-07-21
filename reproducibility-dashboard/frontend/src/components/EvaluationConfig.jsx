import React from 'react';

const EvaluationConfig = ({ formData, handleInputChange, errors }) => {
    return (
        <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Evaluation Configuration</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* Evaluation Metric and @k Value in same row */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Evaluation Metric
                    </label>
                    <div className="flex items-center space-x-6">
                        <label className="flex items-center">
                            <input
                                type="radio"
                                name="evaluation_metric"
                                value="pass@k"
                                checked={formData.evaluation_metric === "pass@k"}
                                onChange={(e) => handleInputChange('evaluation_metric', e.target.value)}
                                className="mr-2"
                            />
                            <span className="text-sm text-gray-700">pass@k</span>
                        </label>
                        <label className="flex items-center">
                            <input
                                type="radio"
                                name="evaluation_metric"
                                value="maj@k"
                                checked={formData.evaluation_metric === "maj@k"}
                                onChange={(e) => handleInputChange('evaluation_metric', e.target.value)}
                                className="mr-2"
                            />
                            <span className="text-sm text-gray-700">maj@k</span>
                        </label>
                    </div>
                    {errors.evaluation_metric && (
                        <p className="mt-1 text-sm text-red-600">{errors.evaluation_metric}</p>
                    )}
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        @k Value
                    </label>
                    <input
                        type="number"
                        value={formData.at_k_value}
                        onChange={(e) => handleInputChange('at_k_value', parseInt(e.target.value) || 1)}
                        min="1"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                    {errors.at_k_value && (
                        <p className="mt-1 text-sm text-red-600">{errors.at_k_value}</p>
                    )}
                    <p className="mt-1 text-xs text-gray-500">
                        The k value for {formData.evaluation_metric} evaluation
                    </p>
                </div>

                {/* Evaluation Tool in its own row */}
                <div className="lg:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Evaluation Tool
                    </label>
                    <div className="flex items-center space-x-6">
                        <label className="flex items-center">
                            <input
                                type="radio"
                                name="evaluation_tool"
                                value="rule-based"
                                checked={formData.evaluation_tool === "rule-based"}
                                onChange={(e) => handleInputChange('evaluation_tool', e.target.value)}
                                className="mr-2"
                            />
                            <span className="text-sm text-gray-700">Rule-based</span>
                        </label>
                        <label className="flex items-center">
                            <input
                                type="radio"
                                name="evaluation_tool"
                                value="llm"
                                checked={formData.evaluation_tool === "llm"}
                                onChange={(e) => handleInputChange('evaluation_tool', e.target.value)}
                                className="mr-2"
                            />
                            <span className="text-sm text-gray-700">LLM Judge</span>
                        </label>
                    </div>
                    {errors.evaluation_tool && (
                        <p className="mt-1 text-sm text-red-600">{errors.evaluation_tool}</p>
                    )}
                </div>

                {/* Rule-based Configuration */}
                {formData.evaluation_tool === "rule-based" && (
                    <div className="lg:col-span-2">
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Extraction Method
                        </label>
                        <div className="flex items-center space-x-6 mb-4">
                            <label className="flex items-center">
                                <input
                                    type="radio"
                                    name="extraction_method"
                                    value="predefined"
                                    checked={formData.extraction_method === "predefined"}
                                    onChange={(e) => handleInputChange('extraction_method', e.target.value)}
                                    className="mr-2"
                                />
                                <span className="text-sm text-gray-700">Predefined Functions</span>
                            </label>
                            <label className="flex items-center">
                                <input
                                    type="radio"
                                    name="extraction_method"
                                    value="custom"
                                    checked={formData.extraction_method === "custom"}
                                    onChange={(e) => handleInputChange('extraction_method', e.target.value)}
                                    className="mr-2"
                                />
                                <span className="text-sm text-gray-700">Custom Code</span>
                            </label>
                        </div>

                        {formData.extraction_method === "predefined" && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Extraction Function
                                </label>
                                <select
                                    value={formData.predefined_extractor || "boxed_answer"}
                                    onChange={(e) => handleInputChange('predefined_extractor', e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                >
                                    <option value="boxed_answer">Boxed Answer (extract text in [box] or [answer])</option>
                                    <option value="last_token">Last Token (extract last word/number)</option>
                                    <option value="answer_tag">Answer Tag (extract text between &lt;answer&gt; tags)</option>
                                    <option value="exact_match">Exact Match (check if answer appears in response)</option>
                                    <option value="contains_answer">Contains Answer (check if response contains answer)</option>
                                </select>
                                <p className="mt-1 text-xs text-gray-500">
                                    Choose a predefined extraction function for rule-based evaluation
                                </p>
                            </div>
                        )}

                        {formData.extraction_method === "custom" && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Custom Extractor Code
                                </label>
                                <textarea
                                    value={formData.custom_extractor_code || ""}
                                    onChange={(e) => handleInputChange('custom_extractor_code', e.target.value)}
                                    rows={6}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                    placeholder="def extract_answer(response, question, answer):
    # Your custom extraction logic here
    # Return True if correct, False otherwise
    return response.lower().find(answer.lower()) != -1"
                                />
                                {errors.custom_extractor_code && (
                                    <p className="mt-1 text-sm text-red-600">{errors.custom_extractor_code}</p>
                                )}
                                <p className="mt-1 text-xs text-gray-500">
                                    Custom Python code for rule-based answer extraction. Define a function that takes response, question, and answer as parameters.
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* LLM Judge Configuration */}
                {formData.evaluation_tool === "llm" && (
                    <div className="lg:col-span-2">
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Judge Model Type
                        </label>
                        <div className="flex items-center space-x-6 mb-4">
                            <label className="flex items-center">
                                <input
                                    type="radio"
                                    name="judge_model_type"
                                    value="api"
                                    checked={formData.judge_model_type === "api"}
                                    onChange={(e) => handleInputChange('judge_model_type', e.target.value)}
                                    className="mr-2"
                                />
                                <span className="text-sm text-gray-700">API Request</span>
                            </label>
                            <label className="flex items-center">
                                <input
                                    type="radio"
                                    name="judge_model_type"
                                    value="local"
                                    checked={formData.judge_model_type === "local"}
                                    onChange={(e) => handleInputChange('judge_model_type', e.target.value)}
                                    className="mr-2"
                                />
                                <span className="text-sm text-gray-700">Local Model</span>
                            </label>
                        </div>

                        {formData.judge_model_type === "api" && (
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Judge Model
                                    </label>
                                    <input
                                        type="text"
                                        value={formData.judge_model || ""}
                                        onChange={(e) => handleInputChange('judge_model', e.target.value)}
                                        placeholder="gpt-4, claude-3-sonnet, llama-2-70b"
                                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                    />
                                    {errors.judge_model && (
                                        <p className="mt-1 text-sm text-red-600">{errors.judge_model}</p>
                                    )}
                                    <p className="mt-1 text-xs text-gray-500">
                                        Model to use as judge for LLM-based evaluation
                                    </p>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Judge API Key
                                    </label>
                                    <input
                                        type="password"
                                        value={formData.judge_api_key || ""}
                                        onChange={(e) => handleInputChange('judge_api_key', e.target.value)}
                                        placeholder="sk-..."
                                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                    />
                                    {errors.judge_api_key && (
                                        <p className="mt-1 text-sm text-red-600">{errors.judge_api_key}</p>
                                    )}
                                    <p className="mt-1 text-xs text-gray-500">
                                        API key for the judge model
                                    </p>
                                </div>
                            </div>
                        )}

                        {formData.judge_model_type === "local" && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Local LLM Path
                                </label>
                                <input
                                    type="text"
                                    value={formData.local_llm_path || ""}
                                    onChange={(e) => handleInputChange('local_llm_path', e.target.value)}
                                    placeholder="/path/to/local/model"
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                />
                                {errors.local_llm_path && (
                                    <p className="mt-1 text-sm text-red-600">{errors.local_llm_path}</p>
                                )}
                                <p className="mt-1 text-xs text-gray-500">
                                    Path to local LLM for judging
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* Evaluation Prompt */}
                <div className="lg:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Evaluation Prompt
                    </label>
                    <textarea
                        value={formData.evaluation_prompt}
                        onChange={(e) => handleInputChange('evaluation_prompt', e.target.value)}
                        rows={4}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        placeholder="Enter your evaluation prompt here. This will be used to guide the evaluation process."
                    />
                    {errors.evaluation_prompt && (
                        <p className="mt-1 text-sm text-red-600">{errors.evaluation_prompt}</p>
                    )}
                    <p className="mt-1 text-xs text-gray-500">
                        Prompt that will be used for evaluation. For rule-based evaluation, this might be a scoring rubric. For LLM evaluation, this guides the judge's assessment.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default EvaluationConfig; 