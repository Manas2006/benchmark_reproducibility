// Configuration
const API_BASE = 'http://localhost:8001';
const WS_BASE = 'ws://localhost:8001';

// Available options
const AVAILABLE_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B',
    'Qwen/Qwen2.5-Math-7B',
    'Qwen/Qwen2.5-Math-14B',
    'Qwen/Qwen2.5-Math-72B',
    'WizardLMTeam/WizardMath-7B-V1.1',
    'TIGER-Lab/MAmmoTH-7B',
    'deepseek-ai/deepseek-math-7b-instruct',
    '01-ai/Yi-1.5-6B-Chat',
    'HuggingFaceTB/SmolLM-135M-Instruct',
    'HuggingFaceTB/SmolLM-1.7B-Instruct',
    'Link from Hugging Face'
];

const AVAILABLE_DATASETS = [
    'gsm8k',
    'math',
    'gsm8k,math'
];

const EVAL_METHODS = [
    'pass@k',
    'maj@k',
    'rm@k'
];

const BACKEND_OPTIONS = ['local', 'slurm'];

// Global state
let modelConfigs = [];
let currentWebSocket = null;
let jobListInterval = null;
let structuredViewMode = true; // true for structured view, false for raw view

// Tab management
function showTab(tabName, event = null) {
    console.log(`Switching to tab: ${tabName}`);

    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('bg-blue-600', 'text-white');
        btn.classList.add('text-gray-500', 'hover:text-gray-700');
    });

    // Show selected tab
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
        console.log(`Tab '${tabName}' activated successfully`);
    } else {
        console.error(`Tab element with id '${tabName}' not found`);
        console.log('Available tab elements:', Array.from(document.querySelectorAll('.tab-content')).map(tab => tab.id));

        // If cot-analysis tab is missing, try to create it dynamically
        if (tabName === 'cot-analysis') {
            console.log('Attempting to create cot-analysis tab dynamically...');
            createCoTAnalysisTab();
            const newTab = document.getElementById(tabName);
            if (newTab) {
                newTab.classList.add('active');
                console.log(`Tab '${tabName}' created and activated successfully`);
            } else {
                console.error('Failed to create cot-analysis tab');
                return;
            }
        } else {
            return;
        }
    }

    // Highlight selected tab button
    if (event && event.target) {
        event.target.classList.remove('text-gray-500', 'hover:text-gray-700');
        event.target.classList.add('bg-blue-600', 'text-white');
    }

    // Start/stop job list auto-refresh
    if (tabName === 'jobs') {
        if (!jobListInterval) {
            refreshJobs();
            jobListInterval = setInterval(refreshJobs, 5000);
        }
    } else {
        if (jobListInterval) {
            clearInterval(jobListInterval);
            jobListInterval = null;
        }
    }

    // Load path config when settings tab is shown
    if (tabName === 'settings') {
        loadPathConfig();
    }
}

// Path configuration functions
async function loadPathConfig() {
    try {
        const response = await fetch(`${API_BASE}/config/paths`);
        const data = await response.json();

        // Populate form fields with current config
        const config = data.current_config;
        document.getElementById('workspace_dir').value = config.workspace_dir || '';
        document.getElementById('evaluation_dir').value = config.evaluation_dir || '';
        document.getElementById('backend_dir').value = config.backend_dir || '';
        document.getElementById('python_path').value = config.python_path || '';
        document.getElementById('conda_env_path').value = config.conda_env_path || '';
        document.getElementById('output_dir').value = config.output_dir || '';
        document.getElementById('logs_dir').value = config.logs_dir || '';
        document.getElementById('scripts_dir').value = config.scripts_dir || '';
        document.getElementById('job_db_path').value = config.job_db_path || '';
        document.getElementById('slurm_partition').value = config.slurm_partition || '';
        document.getElementById('slurm_account').value = config.slurm_account || '';
        document.getElementById('slurm_wall_time').value = config.slurm_wall_time || '';

        console.log('Path configuration loaded:', config);
    } catch (error) {
        console.error('Error loading path configuration:', error);
        alert('Error loading path configuration: ' + error.message);
    }
}

async function savePathConfig(event) {
    event.preventDefault();
    console.log('🔧 savePathConfig called');

    try {
        const config = {
            workspace_dir: document.getElementById('workspace_dir').value,
            evaluation_dir: document.getElementById('evaluation_dir').value,
            backend_dir: document.getElementById('backend_dir').value,
            python_path: document.getElementById('python_path').value,
            conda_env_path: document.getElementById('conda_env_path').value,
            output_dir: document.getElementById('output_dir').value,
            logs_dir: document.getElementById('logs_dir').value,
            scripts_dir: document.getElementById('scripts_dir').value,
            job_db_path: document.getElementById('job_db_path').value,
            slurm_partition: document.getElementById('slurm_partition').value,
            slurm_account: document.getElementById('slurm_account').value,
            slurm_wall_time: document.getElementById('slurm_wall_time').value
        };

        console.log('📤 Sending SLURM config:', {
            partition: config.slurm_partition,
            account: config.slurm_account,
            wall_time: config.slurm_wall_time
        });

        const response = await fetch(`${API_BASE}/config/paths`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();
        if (response.ok) {
            console.log('✅ Path configuration saved successfully:', data);
            alert('Path configuration saved successfully!');
            // Reload the configuration to show updated values
            await loadPathConfig();
        } else {
            console.error('❌ Error saving path configuration:', data);
            alert('Error saving path configuration: ' + (data.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error saving path configuration:', error);
        alert('Error saving path configuration: ' + error.message);
    }
}

async function validatePaths() {
    try {
        const response = await fetch(`${API_BASE}/config/paths/validate`);
        const data = await response.json();

        const resultDiv = document.getElementById('path-validation-result');
        let html = '<div class="p-4 rounded-md ';

        if (data.valid) {
            html += 'bg-green-50 border border-green-200">';
            html += '<h4 class="text-green-800 font-medium">✓ Path validation successful</h4>';
        } else {
            html += 'bg-red-50 border border-red-200">';
            html += '<h4 class="text-red-800 font-medium">✗ Path validation failed</h4>';
        }

        if (data.errors && data.errors.length > 0) {
            html += '<div class="mt-2"><h5 class="text-red-700 font-medium">Errors:</h5><ul class="list-disc list-inside text-red-600 text-sm">';
            data.errors.forEach(error => {
                html += `<li>${error}</li>`;
            });
            html += '</ul></div>';
        }

        if (data.warnings && data.warnings.length > 0) {
            html += '<div class="mt-2"><h5 class="text-yellow-700 font-medium">Warnings:</h5><ul class="list-disc list-inside text-yellow-600 text-sm">';
            data.warnings.forEach(warning => {
                html += `<li>${warning}</li>`;
            });
            html += '</ul></div>';
        }

        html += '</div>';
        resultDiv.innerHTML = html;

    } catch (error) {
        console.error('Error validating paths:', error);
        alert('Error validating paths: ' + error.message);
    }
}

async function resetPathConfig() {
    if (!confirm('Are you sure you want to reset the path configuration to defaults? This will overwrite your current settings.')) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/config/paths/reset`, {
            method: 'POST'
        });

        const data = await response.json();
        if (response.ok) {
            alert('Path configuration reset to defaults successfully!');
            loadPathConfig(); // Reload the form with new defaults
            console.log('Path configuration reset:', data);
        } else {
            alert('Error resetting path configuration: ' + (data.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error resetting path configuration:', error);
        alert('Error resetting path configuration: ' + error.message);
    }
}

// Model configuration management
function addModelConfig() {
    const configId = Date.now();
    const config = {
        id: configId,
        model: AVAILABLE_MODELS[0],
        customModel: '',
        dataset: AVAILABLE_DATASETS[0],
        backend: BACKEND_OPTIONS[0],
        temperature: '0.0',
        top_p: '1.0',
        top_k: '0',
        max_tokens: '2048',
        seed: '42',
        eval_method: EVAL_METHODS[0],
        k: '1',
        prompt: '',
        prompt_type: 'custom',
        enable_prob_tracking: false,
        prob_plot_type: 'aggregate',
        prob_plot_sample_id: ''
    };

    modelConfigs.push(config);
    renderModelConfigs();
}

function removeModelConfig(configId) {
    modelConfigs = modelConfigs.filter(config => config.id !== configId);
    renderModelConfigs();
}

function updateModelConfig(configId, field, value) {
    const config = modelConfigs.find(c => c.id === configId);
    if (config) {
        config[field] = value;
        updateJobCount();
    }
}

function updatePromptField(configId) {
    const config = modelConfigs.find(c => c.id === configId);
    if (!config) return;

    const promptField = document.getElementById(`prompt-field-${configId}`);
    if (!promptField) return;

    if (config.prompt_type === 'custom') {
        // Show custom prompt field
        promptField.style.display = 'block';
        promptField.querySelector('label').textContent = 'Custom Prompt Template';
        promptField.querySelector('textarea').placeholder = 'Enter your custom prompt template here... Use {question} to insert the math problem. Example: \'Solve this math problem step by step: {question}\'';
    } else {
        // Hide custom prompt field for standard prompt types
        promptField.style.display = 'none';
        // Clear the prompt field when switching to standard prompt types
        const textarea = promptField.querySelector('textarea');
        if (textarea) {
            textarea.value = '';
            updateModelConfig(configId, 'prompt', '');
        }
    }
}

function handleModelSelection(configId, selectedValue) {
    const urlInput = document.getElementById(`url-input-${configId}`);
    if (selectedValue === 'Link from Hugging Face') {
        urlInput.classList.remove('hidden');
    } else {
        urlInput.classList.add('hidden');
        // Clear the custom model field when switching away from Link option
        updateModelConfig(configId, 'customModel', '');
    }
}

function toggleViewMode() {
    structuredViewMode = !structuredViewMode;
    const indicator = document.getElementById('view-mode-indicator');
    if (indicator) {
        indicator.textContent = structuredViewMode ? 'Structured View' : 'Raw View';
    }
}

function calculateJobCount() {
    let totalJobs = 0;
    modelConfigs.forEach(config => {
        // Use customModel if it's set (for Link from Hugging Face), otherwise use model
        const models = (config.model === 'Link from Hugging Face' && config.customModel) ? [config.customModel] : [config.model];
        const datasets = config.dataset.split('\n').filter(d => d.trim());
        const temperatures = config.temperature.split('\n').filter(t => t.trim());
        const top_ps = config.top_p.split('\n').filter(t => t.trim());
        const top_ks = config.top_k.split('\n').filter(k => k.trim());
        const seeds = config.seed.split('\n').filter(s => s.trim());
        const ks = config.k.split('\n').filter(k => k.trim());
        const max_tokens = config.max_tokens.split('\n').filter(m => m.trim());

        const combinations = models.length * datasets.length * temperatures.length *
            top_ps.length * top_ks.length * seeds.length * ks.length * max_tokens.length;
        totalJobs += combinations;
    });
    return totalJobs;
}

function updateJobCount() {
    const totalJobs = calculateJobCount();
    const jobCountElement = document.getElementById('job-count');
    if (jobCountElement) {
        jobCountElement.textContent = totalJobs;
        jobCountElement.className = totalJobs > 100 ? 'text-red-600 font-bold' : 'text-blue-600 font-bold';
    }
}

function renderModelConfigs() {
    const container = document.getElementById('model-configs');
    container.innerHTML = '';

    modelConfigs.forEach(config => {
        const configDiv = document.createElement('div');
        configDiv.className = 'border border-gray-200 rounded-lg p-4 mb-4';
        configDiv.innerHTML = `
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-lg font-medium">Model Configuration</h3>
                <button onclick="removeModelConfig(${config.id})" class="text-red-600 hover:text-red-800">
                    Remove
                </button>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700">Model (or Hugging Face URL)</label>
                    <select onchange="updateModelConfig(${config.id}, 'model', this.value); handleModelSelection(${config.id}, this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${AVAILABLE_MODELS.map(model => `<option value="${model}" ${config.model === model ? 'selected' : ''}>${model}</option>`).join('')}
                    </select>
                    <div id="url-input-${config.id}" class="mt-2 ${config.model === 'Link from Hugging Face' ? '' : 'hidden'}">
                        <input type="text" onchange="updateModelConfig(${config.id}, 'customModel', this.value)" 
                               placeholder="Enter Hugging Face URL (e.g., https://huggingface.co/openai/gpt-oss-20b)" 
                               class="block w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                               value="${config.customModel}">
                        <p class="text-xs text-gray-500 mt-1">The model name will be automatically extracted from the URL.</p>
                    </div>
                </div>
                
                <div>
                    <label class="inline-flex items-center mt-6">
                        <input type="checkbox" ${config.enable_prob_tracking ? 'checked' : ''} onchange="updateModelConfig(${config.id}, 'enable_prob_tracking', this.checked)" class="form-checkbox h-5 w-5 text-blue-600">
                        <span class="ml-2 text-sm text-gray-700">Enable probability tracking (requires vLLM)</span>
                    </label>
                    <p class="text-xs text-gray-500 mt-1">If enabled, vLLM will be used and probabilities will be recorded to a separate JSONL.</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Dataset (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'dataset', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="gsm8k&#10;math&#10;gsm8k,math">${config.dataset}</textarea>
                    <p class="text-xs text-gray-500 mt-1">
                        Enter one dataset per line. You can use a built-in name (e.g., <code>gsm8k</code>, <code>math</code>) or a Hugging Face dataset link (e.g., <code>https://huggingface.co/datasets/username/datasetname</code>).
                    </p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Backend</label>
                    <select onchange="updateModelConfig(${config.id}, 'backend', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${BACKEND_OPTIONS.map(backend => `<option value="${backend}" ${config.backend === backend ? 'selected' : ''}>${backend}</option>`).join('')}
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Temperature (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'temperature', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="0.0&#10;0.1&#10;0.5">${config.temperature}</textarea>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Top P (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'top_p', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="1.0&#10;0.9&#10;0.8">${config.top_p}</textarea>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Top K (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'top_k', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="0&#10;10&#10;50">${config.top_k}</textarea>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Seed (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'seed', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="42&#10;123&#10;456">${config.seed}</textarea>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Eval Method</label>
                    <select onchange="updateModelConfig(${config.id}, 'eval_method', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${EVAL_METHODS.map(method => `<option value="${method}" ${config.eval_method === method ? 'selected' : ''}>${method}</option>`).join('')}
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">K for Pass@K (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'k', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="1&#10;2&#10;5">${config.k}</textarea>
                    <p class="text-xs text-gray-500 mt-1">Number of attempts per question for pass@k evaluation</p>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Max Tokens (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'max_tokens', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="2048\n4096">${config.max_tokens}</textarea>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Prompt Type</label>
                    <select onchange="updateModelConfig(${config.id}, 'prompt_type', this.value); updatePromptField(${config.id})" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        <option value="custom" ${config.prompt_type === 'custom' ? 'selected' : ''}>Custom Prompt</option>
                        <option value="cot" ${config.prompt_type === 'cot' ? 'selected' : ''}>Chain of Thought (CoT)</option>
                        <option value="auto-cot" ${config.prompt_type === 'auto-cot' ? 'selected' : ''}>Auto Chain of Thought (Auto-CoT)</option>
                        <option value="pal" ${config.prompt_type === 'pal' ? 'selected' : ''}>Program-aided Language (PAL)</option>
                        <option value="tool-integrated" ${config.prompt_type === 'tool-integrated' ? 'selected' : ''}>Tool Integrated</option>
                        <option value="qwen25-math-cot" ${config.prompt_type === 'qwen25-math-cot' ? 'selected' : ''}>Qwen2.5 Math CoT</option>
                        <option value="direct" ${config.prompt_type === 'direct' ? 'selected' : ''}>Direct</option>
                        <option value="self-instruct" ${config.prompt_type === 'self-instruct' ? 'selected' : ''}>Self Instruct</option>
                        <option value="wizard_zs" ${config.prompt_type === 'wizard_zs' ? 'selected' : ''}>Wizard Zero Shot</option>
                        <option value="platypus_fs" ${config.prompt_type === 'platypus_fs' ? 'selected' : ''}>Platypus Few Shot</option>
                    </select>
                </div>
                
                <div class="md:col-span-2 lg:col-span-3" id="prompt-field-${config.id}">
                    <label class="block text-sm font-medium text-gray-700">Custom Prompt Template</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'prompt', this.value)" 
                              class="mt-1 block w-full border border-blue-300 rounded-md px-3 py-2 bg-blue-50" 
                              rows="6" placeholder="Enter your custom prompt template here... Use {question} to insert the math problem. Example: 'Solve this math problem step by step: {question}'">${config.prompt}</textarea>
                    <p class="text-xs text-blue-600 mt-1 font-medium">
                        ⚡ Use {question} to insert the math problem. Example: "Solve this math problem step by step: {question}"
                    </p>
                    <p class="text-xs text-gray-500 mt-1">
                        💡 Standard prompt types (CoT, PAL, etc.) use pre-built templates with few-shot examples. Custom prompts give you full control over the prompt format.
                    </p>
                </div>
            </div>
        `;
        container.appendChild(configDiv);
        // Set initial state of prompt field
        updatePromptField(config.id);
    });
    updateJobCount();
}

// API functions
async function submitEvaluation() {
    console.log('submitEvaluation called');
    if (modelConfigs.length === 0) {
        alert('Please add at least one model configuration');
        return;
    }

    const totalJobs = calculateJobCount();
    if (totalJobs > 100) {
        if (!confirm(`This will create ${totalJobs} jobs. Are you sure you want to continue?`)) {
            return;
        }
    }

    try {
        const allJobs = [];

        modelConfigs.forEach(config => {
            // Use customModel if it's set (for Link from Hugging Face), otherwise use model
            const models = (config.model === 'Link from Hugging Face' && config.customModel) ? [config.customModel] : [config.model];
            const datasets = config.dataset.split('\n').filter(d => d.trim());
            const temperatures = config.temperature.split('\n').filter(t => t.trim());
            const top_ps = config.top_p.split('\n').filter(t => t.trim());
            const top_ks = config.top_k.split('\n').filter(k => k.trim());
            const seeds = config.seed.split('\n').filter(s => s.trim());
            const ks = config.k.split('\n').filter(k => k.trim());
            const max_tokens = config.max_tokens.split('\n').filter(m => m.trim());

            // Generate all combinations
            models.forEach(model => {
                datasets.forEach(dataset => {
                    temperatures.forEach(temp => {
                        top_ps.forEach(top_p => {
                            top_ks.forEach(top_k => {
                                seeds.forEach(seed => {
                                    ks.forEach(k => {
                                        max_tokens.forEach(max_token => {
                                            // Validate that prompt is provided for custom prompt type
                                            if (config.prompt_type === 'custom' && (!config.prompt || config.prompt.trim() === '')) {
                                                throw new Error(`Custom prompt is required when 'Custom Prompt' is selected for model configuration ${config.id}`);
                                            }

                                            // Validate that URL is provided for Link from Hugging Face
                                            if (config.model === 'Link from Hugging Face' && (!config.customModel || config.customModel.trim() === '')) {
                                                throw new Error(`Hugging Face URL is required when 'Link from Hugging Face' is selected for model configuration ${config.id}`);
                                            }

                                            const requestData = {
                                                model: model,
                                                dataset: dataset,
                                                prompt: config.prompt_type === 'custom' ? config.prompt.trim() : '',
                                                prompt_type: config.prompt_type,
                                                backend: config.backend,
                                                temperature: parseFloat(temp),
                                                top_p: parseFloat(top_p),
                                                top_k: parseInt(top_k),
                                                seed: parseInt(seed),
                                                eval_method: config.eval_method,
                                                k: parseInt(k),
                                                max_tokens: parseInt(max_token),
                                                enable_prob_tracking: !!config.enable_prob_tracking
                                            };
                                            allJobs.push(requestData);
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });

        // Submit all jobs
        const promises = allJobs.map(jobData =>
            fetch(`${API_BASE}/jobs`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(jobData)
            }).then(response => response.json())
        );

        const results = await Promise.all(promises);
        console.log('Submitted jobs:', results);

        // Show success message
        alert(`Successfully submitted ${results.length} evaluation job(s)!`);

        // Switch to jobs tab to see the new jobs
        showTab('jobs');
        refreshJobs();

    } catch (error) {
        console.error('Error submitting evaluation:', error);
        console.error('Error stack:', error.stack);
        alert('Error submitting evaluation: ' + error.message);
    }
}

function renderJobList(jobs) {
    const jobsList = document.getElementById('jobs-list');
    jobsList.innerHTML = '';
    if (jobs.length === 0) {
        jobsList.innerHTML = '<p class="text-gray-500">No jobs found</p>';
        return;
    }
    jobs.forEach(job => {
        const jobDiv = document.createElement('div');
        jobDiv.className = 'border border-gray-200 rounded-lg p-4 mb-4 relative';
        // Always use UUID for job_id, but display SLURM job ID if present
        let jobIdDisplay = job.job_id;
        let slurmIdLine = '';
        if (job.backend === 'slurm' && job.slurm_jid) {
            jobIdDisplay = job.slurm_jid;
            slurmIdLine = `<p class="text-xs text-gray-400">UUID: ${job.job_id}</p>`;
        }
        // Buttons for viewing results
        let resultButtonHtml = '';
        if (job.status === 'DONE' && job.result_file) {
            resultButtonHtml = `
                <div class="flex flex-wrap gap-2 mt-4">
                    <button onclick="showMetricsModal('${job.job_id}', 'Metrics')" class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">View Metrics Inline</button>
                    <button onclick="showResultModal('${job.result_file}', 'Model Outputs')" class="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700">View Model Outputs Inline</button>
                    <button onclick="openCoTAnalysisForJob('${job.job_id}')" class="px-3 py-1 bg-orange-600 text-white text-sm rounded hover:bg-orange-700">🧠 CoT Analysis</button>
                    <button onclick="exportToExcel('${job.job_id}')" class="px-3 py-1 bg-purple-600 text-white text-sm rounded hover:bg-purple-700">Export to Excel</button>
                    ${job.prob_file ? `<button onclick="showResultModal('${job.prob_file}','Probability JSONL')" class="px-3 py-1 bg-teal-600 text-white text-sm rounded hover:bg-teal-700">View Prob JSONL</button>` : ''}
                    ${job.prob_file ? `<button onclick="openProbPlotModal('${job.job_id}')" class="px-3 py-1 bg-pink-600 text-white text-sm rounded hover:bg-pink-700">Plot Probabilities</button>` : ''}
                </div>
            `;
        } else if (job.status !== 'DONE' && job.result_file) {
            // Show disabled buttons for jobs that are not done but have result_file
            resultButtonHtml = `
                <div class="flex space-x-2 mt-4">
                    <button disabled class="px-3 py-1 bg-gray-400 text-white text-sm rounded cursor-not-allowed" title="Results not available yet - job is ${job.status.toLowerCase()}">View Metrics Inline</button>
                    <button disabled class="px-3 py-1 bg-gray-400 text-white text-sm rounded cursor-not-allowed" title="Results not available yet - job is ${job.status.toLowerCase()}">View Model Outputs Inline</button>
                    <button disabled class="px-3 py-1 bg-gray-400 text-white text-sm rounded cursor-not-allowed" title="Results not available yet - job is ${job.status.toLowerCase()}">Export to Excel</button>
                </div>
            `;
        }
        // Delete button always at top right
        // Monitor button only for non-completed jobs
        let monitorButtonHtml = '';
        if (job.status !== 'DONE' && job.status !== 'ERROR') {
            monitorButtonHtml = `<button onclick="monitorJob('${job.job_id}')" class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">Monitor</button>`;
        }
        
        const deleteButtonHtml = `
            <button onclick="deleteJob('${job.job_id}')" title="Delete job" class="px-2 py-1 text-red-600 hover:text-red-800">
                <svg xmlns='http://www.w3.org/2000/svg' class='h-5 w-5' fill='none' viewBox='0 0 24 24' stroke='currentColor'><path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M6 18L18 6M6 6l12 12'/></svg>
            </button>
        `;
        
        jobDiv.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <h3 class="font-medium">Job ID: ${jobIdDisplay}</h3>
                    ${slurmIdLine}
                    <p class="text-sm text-gray-600">Model: ${job.request?.model || 'N/A'}</p>
                    <p class="text-sm text-gray-600">Dataset: ${job.request?.dataset || 'N/A'}</p>
                    <p class="text-sm text-gray-600">Status: <span class="font-medium ${getStatusColor(job.status)}">${job.status}</span></p>
                </div>
                <div class="flex space-x-2 items-start">
                    ${monitorButtonHtml}
                    ${deleteButtonHtml}
                </div>
            </div>
            ${resultButtonHtml}
        `;
        jobsList.appendChild(jobDiv);
    });
}

async function deleteJob(jobId) {
    if (!confirm('Are you sure you want to delete this job?')) return;
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' });
        const data = await response.json();
        if (data.deleted) {
            refreshJobs();
        } else {
            alert('Failed to delete job.');
        }
    } catch (error) {
        alert('Error deleting job: ' + error.message);
    }
}

async function clearAllJobs() {
    try {
        // First get all jobs
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        const jobs = data.jobs || [];
        
        if (jobs.length === 0) {
            alert('No jobs to delete');
            return;
        }
        
        // Confirm deletion
        const confirmation = confirm(`Are you sure you want to delete ALL ${jobs.length} jobs? This action cannot be undone.`);
        if (!confirmation) return;
        
        // Delete all jobs in parallel
        const deletePromises = jobs.map(job => 
            fetch(`${API_BASE}/jobs/${job.job_id}`, { method: 'DELETE' })
        );
        
        // Show progress
        const totalJobs = jobs.length;
        
        // Update progress as jobs are deleted
        const results = await Promise.allSettled(deletePromises);
        
        // Count successful deletions
        let successCount = 0;
        let failCount = 0;
        
        for (const result of results) {
            if (result.status === 'fulfilled') {
                try {
                    const response = result.value;
                    const jobData = await response.json();
                    if (jobData.deleted) {
                        successCount++;
                    } else {
                        failCount++;
                    }
                } catch {
                    failCount++;
                }
            } else {
                failCount++;
            }
        }
        
        // Show results
        if (failCount === 0) {
            alert(`Successfully deleted all ${successCount} jobs!`);
        } else if (successCount === 0) {
            alert(`Failed to delete any jobs. ${failCount} errors occurred.`);
        } else {
            alert(`Deleted ${successCount} jobs successfully. ${failCount} failed to delete.`);
        }
        
        // Refresh the job list
        refreshJobs();
        
    } catch (error) {
        alert('Error clearing jobs: ' + error.message);
    }
}

function getStatusColor(status) {
    switch (status) {
        case 'RUNNING': return 'text-green-600';
        case 'DONE': return 'text-blue-600';
        case 'ERROR': return 'text-red-600';
        case 'READY_FOR_DOWNLOAD': return 'text-yellow-600';
        default: return 'text-gray-600';
    }
}

function renderMonitorControls(jobId) {
    const controlsDiv = document.getElementById('monitor-controls');
    controlsDiv.innerHTML = `<button onclick="cancelJob('${jobId}')" title="Cancel job" class="px-2 py-1 text-red-600 hover:text-red-800">&#10005;</button>`;
}

async function cancelJob(jobId) {
    if (!confirm('Are you sure you want to cancel this job?')) return;
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}/cancel`, { method: 'POST' });
        const data = await response.json();
        if (data.cancelled) {
            alert('Job cancelled.');
            refreshJobs();
        } else {
            alert('Failed to cancel job.');
        }
    } catch (error) {
        alert('Error cancelling job: ' + error.message);
    }
}

function monitorJob(jobId) {
    document.getElementById('monitor-job-id').value = jobId;
    showTab('monitor');
    renderMonitorControls(jobId);
    startMonitoring();
}

function startMonitoring() {
    const jobId = document.getElementById('monitor-job-id').value;
    if (!jobId) {
        alert('Please enter a job ID');
        return;
    }

    // Close existing WebSocket
    if (currentWebSocket) {
        currentWebSocket.close();
    }

    const logOutput = document.getElementById('log-output');
    logOutput.innerHTML = 'Connecting...\n';

    // Connect to WebSocket
    currentWebSocket = new WebSocket(`${WS_BASE}/stream/${jobId}`);

    currentWebSocket.onopen = function () {
        logOutput.innerHTML += 'Connected to job stream\n';
    };

    currentWebSocket.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);
            if (structuredViewMode) {
                // Structured view mode
                if (data.monitor_prompt) {
                    // Display prompt information in a structured way
                    logOutput.innerHTML += `<div style='background: #1a1a1a; border-left: 4px solid #4CAF50; padding: 10px; margin: 10px 0;'>`;
                    logOutput.innerHTML += `<span style='color: #4CAF50; font-weight: bold;'>📝 PROMPT (Sample #${data.monitor_prompt.idx})</span><br>`;
                    logOutput.innerHTML += `<span style='color: #FFD700;'>Question:</span> ${escapeHtml(data.monitor_prompt.question)}<br>`;
                    logOutput.innerHTML += `<span style='color: #FFD700;'>Prompt:</span><br><pre style='color: #87CEEB; white-space: pre-wrap; margin: 5px 0;'>${escapeHtml(data.monitor_prompt.prompt)}</pre>`;
                    logOutput.innerHTML += `</div>`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else if (data.monitor_epoch) {
                    // Display epoch information
                    logOutput.innerHTML += `<div style='background: #1a1a1a; border-left: 4px solid #FF9800; padding: 10px; margin: 10px 0;'>`;
                    logOutput.innerHTML += `<span style='color: #FF9800; font-weight: bold;'>🔄 EPOCH ${data.monitor_epoch.epoch + 1}/${data.monitor_epoch.total_epochs}</span><br>`;
                    logOutput.innerHTML += `<span style='color: #FFD700;'>Remaining prompts:</span> ${data.monitor_epoch.remaining_prompts}`;
                    logOutput.innerHTML += `</div>`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else if (data.monitor_response) {
                    // Display model response (suppressed content)
                    logOutput.innerHTML += `<div style='background: #1a1a1a; border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0;'>`;
                    logOutput.innerHTML += `<span style='color: #2196F3; font-weight: bold;'>🤖 MODEL RESPONSE (Epoch ${data.monitor_response.epoch + 1}, Prompt #${data.monitor_response.prompt_idx})</span><br>`;
                    logOutput.innerHTML += `<span style='color: #FFD700;'>Response:</span> <span style='color: #98FB98;'>[Content suppressed for monitoring]</span>`;
                    logOutput.innerHTML += `</div>`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else if (data.out) {
                    logOutput.innerHTML += `<span style='color: #00ff00;'>[OUT]</span> ${escapeHtml(data.out)}\n`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else if (data.err) {
                    logOutput.innerHTML += `<span style='color: #ff3333;'>[ERR]</span> ${escapeHtml(data.err)}\n`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else if (data.log) {
                    logOutput.innerHTML += data.log + '\n';
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else if (data.gpu) {
                    logOutput.innerHTML += `[GPU] Memory: ${Math.round(data.gpu.mem / 1024 / 1024)}MB, Utilization: ${data.gpu.util}%\n`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else if (data.error) {
                    logOutput.innerHTML += `<span style='color: #ff3333;'>[ERROR]</span> ${escapeHtml(data.error)}\n`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                } else if (data.status) {
                    logOutput.innerHTML += `[STATUS] Job changed to: ${data.status} (Return Code: ${data.return_code || 'N/A'})\n`;
                    logOutput.scrollTop = logOutput.scrollHeight;
                }
            } else {
                // Raw view mode - show everything as plain text
                if (data.out) {
                    logOutput.innerHTML += `<span style='color: #00ff00;'>[OUT]</span> ${escapeHtml(data.out)}\n`;
                } else if (data.err) {
                    logOutput.innerHTML += `<span style='color: #ff3333;'>[ERR]</span> ${escapeHtml(data.err)}\n`;
                } else if (data.log) {
                    logOutput.innerHTML += data.log + '\n';
                } else if (data.gpu) {
                    logOutput.innerHTML += `[GPU] Memory: ${Math.round(data.gpu.mem / 1024 / 1024)}MB, Utilization: ${data.gpu.util}%\n`;
                } else if (data.error) {
                    logOutput.innerHTML += `<span style='color: #ff3333;'>[ERROR]</span> ${escapeHtml(data.error)}\n`;
                } else if (data.status) {
                    logOutput.innerHTML += `[STATUS] Job changed to: ${data.status} (Return Code: ${data.return_code || 'N/A'})\n`;
                } else if (data.monitor_prompt || data.monitor_epoch) {
                    // Show structured data as JSON in raw mode
                    logOutput.innerHTML += `<span style='color: #FFD700;'>[MONITOR]</span> ${escapeHtml(JSON.stringify(data, null, 2))}\n`;
                } else if (data.monitor_response) {
                    // Show model response info without the actual response content
                    const responseInfo = {
                        ...data.monitor_response,
                        response: "[Content suppressed for monitoring]"
                    };
                    logOutput.innerHTML += `<span style='color: #FFD700;'>[MONITOR]</span> ${escapeHtml(JSON.stringify(responseInfo, null, 2))}\n`;
                }
                logOutput.scrollTop = logOutput.scrollHeight;
            }
        } catch (error) {
            logOutput.innerHTML += event.data + '\n';
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    };

    currentWebSocket.onerror = function (error) {
        logOutput.innerHTML += 'WebSocket error: ' + error + '\n';
    };

    currentWebSocket.onclose = function () {
        logOutput.innerHTML += 'WebSocket connection closed\n';
    };
}

function escapeHtml(text) {
    return text.replace(/[&<>"']/g, function (m) {
        switch (m) {
            case '&': return '&amp;';
            case '<': return '&lt;';
            case '>': return '&gt;';
            case '"': return '&quot;';
            case "'": return '&#39;';
            default: return m;
        }
    });
}

async function refreshJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        const jobs = data.jobs || [];
        renderJobList(jobs);
    } catch (error) {
        document.getElementById('jobs-list').innerHTML = '<p class="text-red-500">Error loading jobs</p>';
    }
}

async function showMetricsModal(jobId, title = 'Metrics') {
    try {
        console.log(`Loading metrics for job: ${jobId}`);

        // First, get job configuration information
        let jobConfig = null;
        try {
            const jobResponse = await fetch(`${API_BASE}/jobs/${jobId}`);
            if (jobResponse.ok) {
                jobConfig = await jobResponse.json();
                console.log('Job configuration loaded:', jobConfig);
            } else {
                console.warn(`Failed to fetch job configuration: ${jobResponse.status}`);
            }
        } catch (e) {
            console.warn('Could not fetch job configuration:', e);
        }

        // Then get metrics
        const url = `${API_BASE}/metrics/${jobId}`;
        console.log(`Fetching metrics from: ${url}`);
        const response = await fetch(url);
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            if (response.status === 404) {
                // Try to get more detailed error message from response
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorMessage;
                } catch (e) {
                    // If we can't parse the error response, use the status text
                    errorMessage = `Metrics file not found for job ${jobId}. The job may still be running or may have failed.`;
                }
            }
            throw new Error(errorMessage);
        }
        const content = await response.text();

        // Try to parse metrics as JSON to extract configuration
        let metricsData = null;
        try {
            metricsData = JSON.parse(content);
        } catch (e) {
            console.warn('Could not parse metrics as JSON:', e);
        }

        // Create configuration display
        let configDisplay = '';

        // Try to get configuration from metrics file first, then fallback to job config
        let config = null;
        if (metricsData && metricsData.job_configuration) {
            config = metricsData.job_configuration;
        } else if (jobConfig && jobConfig.request) {
            config = jobConfig.request;
        }

        if (config) {
            configDisplay = `
                <div class="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h4 class="font-semibold text-blue-800 mb-2">Job Configuration:</h4>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div><span class="font-medium">Model:</span> ${config.model || 'N/A'}</div>
                        <div><span class="font-medium">Dataset:</span> ${config.dataset || 'N/A'}</div>
                        <div><span class="font-medium">Temperature:</span> ${config.temperature || 'N/A'}</div>
                        <div><span class="font-medium">Top P:</span> ${config.top_p || 'N/A'}</div>
                        <div><span class="font-medium">Top K:</span> ${config.top_k || 'N/A'}</div>
                        <div><span class="font-medium">Random Seed:</span> ${config.seed || 'N/A'}</div>
                        <div><span class="font-medium">N Sampling:</span> ${config.n_sampling || 'N/A'}</div>
                        <div><span class="font-medium">Max Tokens:</span> ${config.max_tokens || 'N/A'}</div>
                        <div><span class="font-medium">Eval Method:</span> ${config.eval_method || 'N/A'}</div>
                        ${config.prompt_type ? `<div><span class="font-medium">Prompt Type:</span> ${config.prompt_type}</div>` : ''}
                        ${config.k ? `<div><span class="font-medium">K:</span> ${config.k}</div>` : ''}
                    </div>
                    ${config.prompt ? `<div class="mt-2"><span class="font-medium">Custom Prompt:</span><br><code class="text-xs bg-gray-100 p-1 rounded">${escapeHtml(config.prompt)}</code></div>` : ''}
                </div>
            `;
        }

        // Create modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-lg p-6 max-w-4xl max-h-[80vh] overflow-auto">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold">${title}: Job ${jobId}</h3>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 text-xl">&times;</button>
                </div>
                ${configDisplay}
                <div class="mt-4">
                    <h4 class="font-semibold mb-2">Metrics Results:</h4>
                    <pre class="text-sm bg-gray-100 p-4 rounded overflow-auto max-h-64">${escapeHtml(content)}</pre>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e && e.target === modal) {
                modal.remove();
            }
        });
    } catch (error) {
        console.error('Error loading metrics file:', error);
        // Show a more user-friendly error message
        let userMessage = error.message;
        if (error.message.includes('Metrics file not found') || error.message.includes('Job not found')) {
            userMessage = 'The metrics file is not available yet. This could be because:\n\n' +
                '• The job is still running\n' +
                '• The job failed to complete\n' +
                '• The job was cancelled\n\n' +
                'Please check the job status and try again later.';
        }
        alert('Error loading metrics file:\n\n' + userMessage);
    }
}

// Parse CoT from answer field
function parseCoTFromAnswer(answer) {
    if (!answer) return { cot_steps: [], final_answer: '' };

    const raw = answer.trim();

    // Primary heuristic: If the delimiter #### is in raw, split on it
    if (raw.includes('####')) {
        const parts = raw.split('####', 1);
        const cot_text = parts[0].trim();
        const ans_text = parts[1].trim();

        // Convert to structured data
        const cot_steps = cot_text.split('\n').map(line => line.trim()).filter(line => line);
        const final_answer = ans_text;

        return { cot_steps, final_answer };
    } else {
        // Fallback heuristic: Otherwise, split on the last newline
        const lines = raw.split('\n');
        if (lines.length > 1) {
            const cot_text = lines.slice(0, -1).join('\n');
            const ans_text = lines[lines.length - 1];

            const cot_steps = cot_text.split('\n').map(line => line.trim()).filter(line => line);
            const final_answer = ans_text.trim();

            return { cot_steps, final_answer };
        } else {
            return { cot_steps: [], final_answer: raw };
        }
    }
}

async function showResultModal(resultFilePath, title = 'Results') {
    try {
        console.log(`Loading result file: ${resultFilePath}`);
        const url = `${API_BASE}/file?path=${encodeURIComponent(resultFilePath)}`;
        console.log(`Fetching file from: ${url}`);
        const response = await fetch(url);
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            if (response.status === 404) {
                // Try to get more detailed error message from response
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorMessage;
                } catch (e) {
                    // If we can't parse the error response, use the status text
                    errorMessage = `File not found: ${resultFilePath.split('/').pop()}. The job may still be running or may have failed.`;
                }
            }
            throw new Error(errorMessage);
        }
        const content = await response.text();

        // Try to parse as JSON to extract answer field for CoT display
        let jsonData = null;
        let cotDisplay = '';
        try {
            jsonData = JSON.parse(content);
            if (Array.isArray(jsonData) && jsonData.length > 0 && jsonData[0].answer) {
                // Parse CoT from the first sample's answer field
                const firstSample = jsonData[0];
                const cotData = parseCoTFromAnswer(firstSample.answer);

                if (cotData.cot_steps.length > 0) {
                    cotDisplay = `
                        <div class="mb-4 p-4 bg-blue-50 rounded-lg">
                            <h4 class="font-semibold text-blue-800 mb-2">Chain of Thought Analysis:</h4>
                            <div class="mb-2">
                                <strong>Question:</strong> ${escapeHtml(firstSample.question || 'N/A')}
                            </div>
                            <div class="mb-2">
                                <strong>Reasoning Steps:</strong>
                                <ol class="list-decimal list-inside ml-4">
                                    ${cotData.cot_steps.map(step => `<li class="mb-1">${escapeHtml(step)}</li>`).join('')}
                                </ol>
                            </div>
                            <div class="mb-2">
                                <strong>Final Answer:</strong> <span class="font-mono bg-yellow-100 px-2 py-1 rounded">${escapeHtml(cotData.final_answer)}</span>
                            </div>
                            <div class="mb-2">
                                <strong>Expected Answer:</strong> <span class="font-mono bg-green-100 px-2 py-1 rounded">${escapeHtml(firstSample.gt || 'N/A')}</span>
                            </div>
                        </div>
                    `;
                }
            }
        } catch (e) {
            // Not JSON or parsing failed, continue with raw display
        }

        // Create modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-lg p-6 max-w-4xl max-h-96 overflow-auto">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold">${title}: ${resultFilePath.split('/').pop()}</h3>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 text-xl">&times;</button>
                </div>
                ${cotDisplay}
                <div class="mt-4">
                    <h4 class="font-semibold mb-2">Raw JSON Data:</h4>
                    <pre class="text-sm bg-gray-100 p-4 rounded overflow-auto max-h-64">${escapeHtml(content)}</pre>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e && e.target === modal) {
                modal.remove();
            }
        });
    } catch (error) {
        console.error('Error loading result file:', error);
        // Show a more user-friendly error message
        let userMessage = error.message;
        if (error.message.includes('File not found') || error.message.includes('Result file not found')) {
            userMessage = 'The result file is not available yet. This could be because:\n\n' +
                '• The job is still running\n' +
                '• The job failed to complete\n' +
                '• The job was cancelled\n\n' +
                'Please check the job status and try again later.';
        }
        alert('Error loading result file:\n\n' + userMessage);
    }
}

function openProbPlotModal(jobId) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-lg p-6 w-full max-w-xl">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-lg font-semibold">Plot Probabilities</h3>
                <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 text-xl">&times;</button>
            </div>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700">Plot Type</label>
                    <select id="prob-plot-type" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        <option value="aggregate" selected>Aggregate</option>
                        <option value="single">Single</option>
                    </select>
                </div>
                <div id="prob-sample-id-field" class="hidden">
                    <label class="block text-sm font-medium text-gray-700">Sample ID (idx)</label>
                    <input id="prob-sample-id" type="number" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" placeholder="Enter sample idx">
                </div>
                <div class="flex justify-end gap-2">
                    <button id="prob-plot-run" class="px-4 py-2 bg-pink-600 text-white rounded hover:bg-pink-700">Generate Plot</button>
                </div>
                <div id="prob-plot-output" class="mt-4"></div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    const plotTypeSel = modal.querySelector('#prob-plot-type');
    const sampleField = modal.querySelector('#prob-sample-id-field');
    plotTypeSel.addEventListener('change', () => {
        if (plotTypeSel.value === 'single') sampleField.classList.remove('hidden');
        else sampleField.classList.add('hidden');
    });
    modal.querySelector('#prob-plot-run').addEventListener('click', async () => {
        const plotType = plotTypeSel.value;
        const sampleIdVal = modal.querySelector('#prob-sample-id').value;
        const params = new URLSearchParams();
        params.append('plot_type', plotType);
        if (plotType === 'single') {
            if (!sampleIdVal) {
                alert('Please enter a sample idx for single plot');
                return;
            }
            params.append('sample_id', String(parseInt(sampleIdVal)));
        }
        const imgContainer = modal.querySelector('#prob-plot-output');
        imgContainer.innerHTML = 'Generating plot...';
        try {
            const url = `${API_BASE}/jobs/${jobId}/prob-plot?${params.toString()}`;
            const resp = await fetch(url);
            if (!resp.ok) {
                const err = await resp.text();
                throw new Error(err || `HTTP ${resp.status}`);
            }
            const blob = await resp.blob();
            const imgUrl = URL.createObjectURL(blob);
            imgContainer.innerHTML = `<img src="${imgUrl}" class="max-h-[60vh] rounded border"/>`;
        } catch (e) {
            imgContainer.innerHTML = `<div class="text-red-600">Error: ${escapeHtml(e.message || String(e))}</div>`;
        }
    });
    // Close on background click
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });
}

async function exportToExcel(jobId) {
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}/export`);
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            if (response.status === 404) {
                errorMessage = 'Job not found or no results available yet.';
            } else if (response.status === 500) {
                errorMessage = 'Error generating Excel file. The job may still be running or may have failed.';
            }
            throw new Error(errorMessage);
        }

        // Get the filename from the response headers
        const contentDisposition = response.headers.get('content-disposition');
        let filename = 'results.xlsx';
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="(.+)"/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Create blob and download
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

    } catch (error) {
        console.error('Error exporting to Excel:', error);
        alert('Error exporting to Excel:\n\n' + error.message);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    // Add initial model config
    addModelConfig();

    // Load jobs on page load
    refreshJobs();
    // Add monitor controls container
    const monitorTab = document.getElementById('monitor');
    const controlsDiv = document.createElement('div');
    controlsDiv.id = 'monitor-controls';
    controlsDiv.className = 'mb-2 flex justify-end';
    monitorTab.insertBefore(controlsDiv, monitorTab.children[1]);
    // Start auto-refresh if jobs tab is active on load
    if (document.getElementById('jobs').classList.contains('active')) {
        jobListInterval = setInterval(refreshJobs, 5000);
    }

    // Add form submit handler for path configuration
    document.getElementById('path-config-form').addEventListener('submit', savePathConfig);

    // Add CoT job select change handler
    document.getElementById('cot-job-select').addEventListener('change', function () {
        const analyzeBtn = document.getElementById('analyze-btn');
        const exportExcelBtn = document.getElementById('export-excel-btn');
        const exportJsonBtn = document.getElementById('export-json-btn');

        if (this.value) {
            analyzeBtn.disabled = false;
            exportExcelBtn.disabled = false;
            exportJsonBtn.disabled = false;
        } else {
            analyzeBtn.disabled = true;
            exportExcelBtn.disabled = true;
            exportJsonBtn.disabled = true;
        }
    });

    // Load CoT jobs when page loads
    loadCoTJobs();
});

// ===== CoT ANALYSIS FUNCTIONS =====

let currentCoTData = null; // Store current analysis data

async function loadCoTJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        const jobs = data.jobs || [];

        // Filter for completed jobs
        const completedJobs = jobs.filter(job => job.status === 'DONE' && job.result_file);

        const select = document.getElementById('cot-job-select');
        select.innerHTML = '<option value="">Select a completed job...</option>';

        completedJobs.forEach(job => {
            const option = document.createElement('option');
            option.value = job.job_id;

            // Display SLURM ID if available, otherwise UUID
            let displayName = job.job_id;
            if (job.backend === 'slurm' && job.slurm_jid) {
                displayName = `${job.slurm_jid} (${job.job_id.substring(0, 8)}...)`;
            }

            // Extract model name from the request field, format it nicely
            let modelName = 'Unknown Model';
            if (job.request && job.request.model) {
                modelName = job.request.model;
                // If it's a HuggingFace path, show just the model name part
                if (modelName.includes('/')) {
                    modelName = modelName.split('/').slice(-1)[0];
                }
            }

            option.textContent = `${displayName} - ${modelName}`;
            select.appendChild(option);
        });

        console.log(`Loaded ${completedJobs.length} completed jobs for CoT analysis`);
    } catch (error) {
        console.error('Error loading jobs for CoT analysis:', error);
        showCoTError('Failed to load jobs for analysis');
    }
}

async function runCoTAnalysis() {
    const jobSelect = document.getElementById('cot-job-select');
    if (!jobSelect) {
        console.error('cot-job-select element not found');
        showCoTError('CoT Analysis tab not properly loaded. Please try refreshing the page.');
        return;
    }

    const jobId = jobSelect.value;
    if (!jobId) {
        showCoTError('Please select a job to analyze');
        return;
    }

    // Show loading state
    showCoTLoading();
    hideCoTError();
    hideCoTResults();

    try {
        console.log(`Running CoT analysis for job: ${jobId}`);

        // Call the backend CoT analysis endpoint
        const response = await fetch(`${API_BASE}/jobs/${jobId}/cot-analysis`);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }

        const analysisData = await response.json();
        console.log('API response received:', analysisData);
        console.log('Setting currentCoTData...');
        currentCoTData = analysisData;
        console.log('currentCoTData set successfully');

        // Hide loading and show results
        console.log('About to hide loading...');
        hideCoTLoading();
        console.log('Loading hidden, about to show results...');
        showCoTResults(analysisData);
        console.log('showCoTResults call completed');

    } catch (error) {
        console.error('Error running CoT analysis:', error);
        hideCoTLoading();
        showCoTError(`Analysis failed: ${error.message}`);
    }
}

function showCoTResults(data) {
    try {
        console.log('CoT Analysis Data received:', data);
        console.log('Job summary:', data.job_summary);
        console.log('Per sample metrics length:', data.per_sample_metrics?.length);

        console.log('About to show results section...');
        document.getElementById('cot-analysis-results').classList.remove('hidden');
        console.log('Results section shown');

        // Populate summary statistics
        console.log('About to call populateCoTSummary...');
        populateCoTSummary(data.job_summary);
        console.log('populateCoTSummary completed');

        // Populate CQS component scores
        console.log('About to call populateCQSComponents...');
        populateCQSComponents(data.job_summary);
        console.log('populateCQSComponents completed');

        // Show random samples by default
        console.log('About to call showRandomSamples...');
        showRandomSamples();
        console.log('showRandomSamples completed');
    } catch (error) {
        console.error('ERROR in showCoTResults:', error);
        console.error('Error stack:', error.stack);
    }
}

function populateCoTSummary(summary) {
    console.log('Populating CoT Summary with:', summary);
    const summaryDiv = document.getElementById('cot-summary');

    if (!summaryDiv) {
        console.error('ERROR: cot-summary element not found!');
        return;
    }

    if (!summary) {
        console.error('Summary is undefined');
        summaryDiv.innerHTML = '<div class="text-red-500">Error: No summary data</div>';
        return;
    }

    console.log('About to populate summary with total_samples:', summary.total_samples);

    try {
        summaryDiv.innerHTML = `
            <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">${summary.total_samples || 0}</div>
                <div class="text-sm text-gray-600">Total Samples</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">${(summary.cqs_score_avg || 0).toFixed(3)}</div>
                <div class="text-sm text-gray-600">Average CQS Score</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">${summary.samples_with_reasoning || 0}</div>
                <div class="text-sm text-gray-600">With Reasoning</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">${(summary.avg_reasoning_steps || 0).toFixed(1)}</div>
                <div class="text-sm text-gray-600">Avg Steps</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">${(summary.avg_reasoning_length || 0).toFixed(0)}</div>
                <div class="text-sm text-gray-600">Avg Length (chars)</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">${getScoreInterpretation(summary.cqs_score_avg || 0)}</div>
                <div class="text-sm text-gray-600">Overall Quality</div>
            </div>
        `;
        console.log('CoT Summary populated successfully');
    } catch (error) {
        console.error('Error populating CoT summary:', error);
        summaryDiv.innerHTML = '<div class="text-red-500">Error populating summary</div>';
    }
}

function populateCQSComponents(summary) {
    console.log('Populating CQS Components with:', summary);
    const componentsDiv = document.getElementById('cqs-components');

    if (!componentsDiv) {
        console.error('ERROR: cqs-components element not found!');
        return;
    }

    if (!summary) {
        console.error('Summary is undefined for CQS components');
        componentsDiv.innerHTML = '<div class="text-red-500">Error: No summary data for components</div>';
        return;
    }

    console.log('Populating CQS components with final_answer_correctness_avg:', summary.final_answer_correctness_avg);

    try {
        const components = [
            { name: 'Final Answer Correctness', score: summary.final_answer_correctness_avg || 0, weight: '30%', color: 'text-red-600' },
            { name: 'Arithmetic Accuracy', score: summary.arithmetic_accuracy_avg || 0, weight: '25%', color: 'text-orange-600' },
            { name: 'Logical Structure', score: summary.logical_structure_avg || 0, weight: '20%', color: 'text-yellow-600' },
            { name: 'Consistency & Completeness', score: summary.consistency_completeness_avg || 0, weight: '15%', color: 'text-green-600' },
            { name: 'Formatting & Notation', score: summary.formatting_notation_avg || 0, weight: '10%', color: 'text-blue-600' }
        ];

        console.log('CQS Components:', components);

        componentsDiv.innerHTML = components.map(comp => `
            <div class="text-center p-3 bg-white rounded-lg border">
                <div class="text-lg font-bold ${comp.color}">${comp.score.toFixed(3)}</div>
                <div class="text-sm font-medium text-gray-800">${comp.name}</div>
                <div class="text-xs text-gray-500">${comp.weight} weight</div>
                <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div class="bg-gradient-to-r from-red-400 to-green-400 h-2 rounded-full" 
                         style="width: ${Math.max(0, Math.min(100, comp.score * 100))}%"></div>
                </div>
            </div>
        `).join('');

        console.log('CQS Components populated successfully');
    } catch (error) {
        console.error('Error populating CQS components:', error);
        componentsDiv.innerHTML = '<div class="text-red-500">Error populating components</div>';
    }
}

function showTopPerformers() {
    if (!currentCoTData) return;

    const samples = currentCoTData.per_sample_metrics
        .sort((a, b) => b.metrics.cqs_score - a.metrics.cqs_score)
        .slice(0, 10);

    renderSampleAnalysis(samples, 'Top 10 Performers (Highest CQS Scores)');
}

function showBottomPerformers() {
    if (!currentCoTData) return;

    const samples = currentCoTData.per_sample_metrics
        .sort((a, b) => a.metrics.cqs_score - b.metrics.cqs_score)
        .slice(0, 10);

    renderSampleAnalysis(samples, 'Bottom 10 Performers (Lowest CQS Scores)');
}

function showRandomSamples() {
    console.log('ShowRandomSamples called, currentCoTData:', currentCoTData);

    if (!currentCoTData) {
        console.error('No currentCoTData available for samples');
        return;
    }

    if (!currentCoTData.per_sample_metrics) {
        console.error('No per_sample_metrics in currentCoTData');
        return;
    }

    console.log('Available samples:', currentCoTData.per_sample_metrics.length);

    const samples = [...currentCoTData.per_sample_metrics]
        .sort(() => 0.5 - Math.random())
        .slice(0, 10);

    console.log('Selected samples for display:', samples.length);
    renderSampleAnalysis(samples, 'Random Sample Selection');
}

function renderSampleAnalysis(samples, title) {
    const sampleDiv = document.getElementById('sample-analysis');

    sampleDiv.innerHTML = `
        <h4 class="font-semibold text-gray-800 mb-3">${title}</h4>
        <div class="space-y-3">
            ${samples.map(sample => `
                <div class="border border-gray-200 rounded-lg p-3 bg-white">
                    <div class="flex justify-between items-start mb-2">
                        <span class="text-sm font-medium text-gray-700">Sample #${sample.idx}</span>
                        <span class="text-sm font-bold ${getCQSColorClass(sample.metrics.cqs_score)}"">
                            CQS: ${sample.metrics.cqs_score.toFixed(3)}
                        </span>
                    </div>
                    
                    <div class="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs mb-2">
                        <div>Final: ${sample.metrics.final_answer_correctness.toFixed(2)}</div>
                        <div>Arith: ${sample.metrics.arithmetic_accuracy.toFixed(2)}</div>
                        <div>Logic: ${sample.metrics.logical_structure_score.toFixed(2)}</div>
                        <div>Consist: ${sample.metrics.consistency_completeness.toFixed(2)}</div>
                        <div>Format: ${sample.metrics.formatting_notation.toFixed(2)}</div>
                    </div>
                    
                    <div class="text-xs text-gray-600">
                        Steps: ${sample.metrics.reasoning_steps} | 
                        Length: ${sample.metrics.total_chars} chars | 
                        Correct: ${sample.is_correct ? '✅' : '❌'}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function getScoreInterpretation(score) {
    if (score >= 0.85) return '🌟 Excellent';
    if (score >= 0.70) return '👍 Good';
    if (score >= 0.50) return '⚠️ Fair';
    return '❌ Poor';
}

function getCQSColorClass(score) {
    if (score >= 0.85) return 'text-green-600';
    if (score >= 0.70) return 'text-blue-600';
    if (score >= 0.50) return 'text-yellow-600';
    return 'text-red-600';
}

function openCoTAnalysisForJob(jobId) {
    try {
        console.log(`Opening CoT analysis for job: ${jobId}`);

        // Switch to CoT Analysis tab
        showTab('cot-analysis');

        // Load jobs for the dropdown first, then wait for elements
        loadCoTJobs().then(() => {
            // Wait for the cot-job-select element to be available
            waitForElement('#cot-job-select').then((jobSelect) => {
                // Select the job
                jobSelect.value = jobId;
                console.log(`Selected job ${jobId} in dropdown`);

                // Enable the analyze button
                const analyzeBtn = document.getElementById('analyze-btn');
                const exportExcelBtn = document.getElementById('export-excel-btn');
                const exportJsonBtn = document.getElementById('export-json-btn');

                if (analyzeBtn) {
                    analyzeBtn.disabled = false;
                    console.log('Analyze button enabled');
                }
                if (exportExcelBtn) exportExcelBtn.disabled = false;
                if (exportJsonBtn) exportJsonBtn.disabled = false;

                // Optionally run analysis immediately
                setTimeout(() => {
                    if (confirm('Run CoT analysis for this job now?')) {
                        runCoTAnalysis();
                    }
                }, 100);
            }).catch(error => {
                console.error('Error waiting for cot-job-select element:', error);
                alert('CoT Analysis tab not properly loaded. Please try refreshing the page.');
            });
        }).catch(error => {
            console.error('Error loading CoT jobs:', error);
            alert('Error loading jobs for CoT analysis: ' + error.message);
        });
    } catch (error) {
        console.error('Error in openCoTAnalysisForJob:', error);
        alert('Error opening CoT analysis: ' + error.message);
    }
}

async function exportCoTExcel() {
    const jobId = document.getElementById('cot-job-select').value;
    if (!jobId) {
        showCoTError('Please select a job first');
        return;
    }

    try {
        // Use the existing Excel export functionality
        exportToExcel(jobId);
    } catch (error) {
        console.error('Error exporting to Excel:', error);
        showCoTError('Failed to export Excel file');
    }
}

async function exportCoTJSON() {
    if (!currentCoTData) {
        showCoTError('Please run analysis first');
        return;
    }

    try {
        // Create and download JSON file
        const dataStr = JSON.stringify(currentCoTData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });

        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `cot_analysis_${currentCoTData.job_id}.json`;
        link.click();

    } catch (error) {
        console.error('Error exporting JSON:', error);
        showCoTError('Failed to export JSON file');
    }
}

// Helper functions for showing/hiding UI states
function showCoTLoading() {
    document.getElementById('cot-loading').classList.remove('hidden');
}

function hideCoTLoading() {
    document.getElementById('cot-loading').classList.add('hidden');
}

function showCoTError(message) {
    document.getElementById('cot-error-message').textContent = message;
    document.getElementById('cot-error').classList.remove('hidden');
}

function hideCoTError() {
    document.getElementById('cot-error').classList.add('hidden');
}



function hideCoTResults() {
    document.getElementById('cot-analysis-results').classList.add('hidden');
}

// Function to create the CoT Analysis tab dynamically
function createCoTAnalysisTab() {
    const mainContent = document.querySelector('main');
    if (!mainContent) {
        console.error('Main content area not found');
        return;
    }

    const cotAnalysisHTML = `
        <div id="cot-analysis" class="tab-content">
            <div class="bg-white shadow rounded-lg p-6">
                <h2 class="text-xl font-semibold mb-4">Chain-of-Thought Analysis</h2>
                <p class="text-sm text-gray-600 mb-6">
                    Analyze the reasoning quality of completed evaluation jobs using our rigorous CoT Quality Score (CQS) system.
                </p>

                <!-- Job Selection -->
                <div class="mb-6">
                    <label for="cot-job-select" class="block text-sm font-medium text-gray-700 mb-2">
                        Select Job for Analysis
                    </label>
                    <select id="cot-job-select" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                        <option value="">Select a completed job...</option>
                    </select>
                </div>

                <!-- Analysis Controls -->
                <div class="flex space-x-4 mb-6">
                    <button onclick="runCoTAnalysis()" id="analyze-btn" disabled
                        class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed">
                        🧠 Run CoT Analysis
                    </button>
                    <button onclick="loadCoTJobs()"
                        class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700">
                        🔄 Refresh Jobs
                    </button>
                </div>

                <!-- Analysis Results -->
                <div id="cot-analysis-results" class="hidden">
                    <!-- Job Summary -->
                    <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                        <h3 class="text-lg font-semibold text-blue-900 mb-3">📊 Overall CQS Summary</h3>
                        <div id="cot-summary" class="grid grid-cols-2 md:grid-cols-3 gap-4">
                            <!-- Summary stats will be populated here -->
                        </div>
                    </div>

                    <!-- CQS Component Breakdown -->
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-6">
                        <h3 class="text-lg font-semibold text-gray-900 mb-3">🏆 CQS Component Scores</h3>
                        <div id="cqs-components" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            <!-- CQS components will be populated here -->
                        </div>
                    </div>

                    <!-- Sample Analysis -->
                    <div class="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
                        <h3 class="text-lg font-semibold text-green-900 mb-3">🔍 Sample Analysis</h3>
                        <div class="flex space-x-4 mb-4">
                            <button onclick="showTopPerformers()" 
                                class="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700">
                                Top Performers
                            </button>
                            <button onclick="showBottomPerformers()" 
                                class="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700">
                                Bottom Performers
                            </button>
                            <button onclick="showRandomSamples()" 
                                class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">
                                Random Samples
                            </button>
                        </div>
                        <div id="sample-analysis" class="max-h-96 overflow-y-auto">
                            <!-- Sample details will be populated here -->
                        </div>
                    </div>

                    <!-- Export Options -->
                    <div class="border-t pt-4">
                        <h3 class="text-lg font-semibold text-gray-900 mb-3">📥 Export Analysis</h3>
                        <div class="flex space-x-4">
                            <button onclick="exportCoTExcel()" id="export-excel-btn" disabled
                                class="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed">
                                📊 Export Enhanced Excel
                            </button>
                            <button onclick="exportCoTJSON()" id="export-json-btn" disabled
                                class="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed">
                                📋 Export JSON Data
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Loading State -->
                <div id="cot-loading" class="hidden text-center py-8">
                    <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    <p class="mt-2 text-gray-600">Analyzing Chain-of-Thought reasoning...</p>
                </div>

                <!-- Error State -->
                <div id="cot-error" class="hidden bg-red-50 border border-red-200 rounded-lg p-4">
                    <h3 class="text-red-800 font-semibold">Analysis Error</h3>
                    <p id="cot-error-message" class="text-red-700"></p>
                </div>
            </div>
        </div>
    `;

    mainContent.insertAdjacentHTML('beforeend', cotAnalysisHTML);
    console.log('CoT Analysis tab created dynamically');
}

// Helper function to wait for an element to be available
function waitForElement(selector, timeout = 5000) {
    return new Promise((resolve, reject) => {
        const element = document.querySelector(selector);
        if (element) {
            resolve(element);
            return;
        }

        const observer = new MutationObserver((mutations, obs) => {
            const element = document.querySelector(selector);
            if (element) {
                obs.disconnect();
                resolve(element);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        setTimeout(() => {
            observer.disconnect();
            reject(new Error(`Element ${selector} not found within ${timeout}ms`));
        }, timeout);
    });
}

// Ensure DOM is loaded before running any functions
document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM loaded, initializing UI...');

    // Verify that all expected tab elements exist
    const expectedTabs = ['configure', 'monitor', 'jobs', 'cot-analysis', 'settings'];
    let missingTabs = [];

    expectedTabs.forEach(tabId => {
        const tab = document.getElementById(tabId);
        if (tab) {
            console.log(`✓ Tab '${tabId}' found`);
        } else {
            console.error(`✗ Tab '${tabId}' NOT found`);
            missingTabs.push(tabId);
        }
    });

    // If cot-analysis tab is missing, create it dynamically
    if (missingTabs.includes('cot-analysis')) {
        console.log('Creating missing cot-analysis tab...');
        createCoTAnalysisTab();
    }

    // Debug: Check if the HTML content is complete
    const mainContent = document.querySelector('main');
    if (mainContent) {
        console.log('Main content found, children count:', mainContent.children.length);
        console.log('Main content HTML length:', mainContent.innerHTML.length);
    } else {
        console.error('Main content not found!');
    }

    // Initialize any necessary UI elements here
});