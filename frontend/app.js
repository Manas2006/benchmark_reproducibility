// Configuration
const API_BASE = 'http://localhost:8002';
const WS_BASE = 'ws://localhost:8002';

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
const INFERENCE_MODES = ['local', 'together_api'];

const TOGETHER_API_MODELS = [
    'deepseek-ai/DeepSeek-R1-0528-tput',
    'deepseek-ai/DeepSeek-R1-0528',
    'deepseek-ai/DeepSeek-R1-0528-Instruct',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-tput',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-v2',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-v2-tput',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-v2.5',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-v2.5-tput',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-v2.5-32k',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-v2.5-32k-tput',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-v2.5-128k',
    'deepseek-ai/DeepSeek-R1-0528-Instruct-v2.5-128k-tput',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.1-70B-Instruct',
    'meta-llama/Llama-3.1-405B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct-tput',
    'meta-llama/Llama-3.1-70B-Instruct-tput',
    'meta-llama/Llama-3.1-405B-Instruct-tput',
    'meta-llama/Llama-3.1-8B-Instruct-32k',
    'meta-llama/Llama-3.1-70B-Instruct-32k',
    'meta-llama/Llama-3.1-405B-Instruct-32k',
    'meta-llama/Llama-3.1-8B-Instruct-32k-tput',
    'meta-llama/Llama-3.1-70B-Instruct-32k-tput',
    'meta-llama/Llama-3.1-405B-Instruct-32k-tput',
    'meta-llama/Llama-3.1-8B-Instruct-128k',
    'meta-llama/Llama-3.1-70B-Instruct-128k',
    'meta-llama/Llama-3.1-405B-Instruct-128k',
    'meta-llama/Llama-3.1-8B-Instruct-128k-tput',
    'meta-llama/Llama-3.1-70B-Instruct-128k-tput',
    'meta-llama/Llama-3.1-405B-Instruct-128k-tput',
    'Qwen/Qwen2.5-Math-1.5B',
    'Qwen/Qwen2.5-Math-7B',
    'Qwen/Qwen2.5-Math-14B',
    'Qwen/Qwen2.5-Math-72B',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
    'Qwen/Qwen2.5-32B-Instruct',
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen2.5-110B-Instruct',
    'Qwen/Qwen2.5-110B-Instruct-tput',
    'Qwen/Qwen2.5-110B-Instruct-32k',
    'Qwen/Qwen2.5-110B-Instruct-32k-tput',
    'Qwen/Qwen2.5-110B-Instruct-128k',
    'Qwen/Qwen2.5-110B-Instruct-128k-tput',
    'Custom Together Model'
];

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
    
    // Load jobs for specific tabs
    if (tabName === 'truncation-analysis') {
        loadTruncationJobs();
    } else if (tabName === 'heatmap') {
        loadHeatmapJobs();
    }
}

// Path configuration functions
async function loadPathConfig() {
    try {
        const response = await fetch(`${API_BASE}/config/paths`);
        const data = await response.json();

        // Populate form fields with current config
        const config = data.current_config;
        const setValue = (id, value) => {
            const el = document.getElementById(id);
            if (el) {
                el.value = value || '';
            } else {
                console.warn(`Element with id '${id}' not found in DOM`);
            }
        };
        
        setValue('workspace_dir', config.workspace_dir);
        setValue('evaluation_dir', config.evaluation_dir);
        setValue('backend_dir', config.backend_dir);
        setValue('python_path', config.python_path);
        setValue('conda_env_path', config.conda_env_path);
        setValue('output_dir', config.output_dir);
        setValue('exports_dir', config.exports_dir);
        setValue('logs_dir', config.logs_dir);
        setValue('scripts_dir', config.scripts_dir);
        setValue('job_db_path', config.job_db_path);
        setValue('slurm_partition', config.slurm_partition);
        setValue('slurm_account', config.slurm_account);
        setValue('slurm_wall_time', config.slurm_wall_time);
        setValue('openai_api_key', config.openai_api_key);
        setValue('hf_token', config.hf_token);

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
            exports_dir: document.getElementById('exports_dir').value,
            logs_dir: document.getElementById('logs_dir').value,
            scripts_dir: document.getElementById('scripts_dir').value,
            job_db_path: document.getElementById('job_db_path').value,
            slurm_partition: document.getElementById('slurm_partition').value,
            slurm_account: document.getElementById('slurm_account').value,
            slurm_wall_time: document.getElementById('slurm_wall_time').value,
            openai_api_key: document.getElementById('openai_api_key').value,
            hf_token: document.getElementById('hf_token').value
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
        inference_mode: 'local',
        together_model: TOGETHER_API_MODELS[0],
        together_custom_model: '',
        together_api_key: '',
        together_logprobs: '0',
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
        enable_path_vectors: false,
        max_path_steps: '0',
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

function handleTogetherModelSelection(configId, selectedValue) {
    const customInput = document.getElementById(`together-custom-input-${configId}`);
    if (selectedValue === 'Custom Together Model') {
        customInput.classList.remove('hidden');
    } else {
        customInput.classList.add('hidden');
        // Clear the custom model field when switching away from Custom option
        updateModelConfig(configId, 'together_custom_model', '');
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
        let models = [];
        if (config.inference_mode === 'together_api') {
            if (config.together_model === 'Custom Together Model' && config.together_custom_model) {
                models = [config.together_custom_model];
            } else if (config.together_model !== 'Custom Together Model') {
                models = [config.together_model];
            } else {
                models = []; // No valid model selected
            }
        } else {
            models = (config.model === 'Link from Hugging Face' && config.customModel) ? [config.customModel] : [config.model];
        }

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
                ${config.inference_mode !== 'together_api' ? `
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
                ` : ''}
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Inference</label>
                    <select onchange="updateModelConfig(${config.id}, 'inference_mode', this.value); renderModelConfigs()" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${INFERENCE_MODES.map(m => `<option value="${m}" ${config.inference_mode === m ? 'selected' : ''}>${m}</option>`).join('')}
                    </select>
                </div>

                ${config.inference_mode === 'together_api' ? `
                <div>
                    <label class="block text-sm font-medium text-gray-700">Together API Model</label>
                    <select onchange="updateModelConfig(${config.id}, 'together_model', this.value); handleTogetherModelSelection(${config.id}, this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${TOGETHER_API_MODELS.map(model => `<option value="${model}" ${config.together_model === model ? 'selected' : ''}>${model}</option>`).join('')}
                    </select>
                    <div id="together-custom-input-${config.id}" class="mt-2 ${config.together_model === 'Custom Together Model' ? '' : 'hidden'}">
                        <input type="text" onchange="updateModelConfig(${config.id}, 'together_custom_model', this.value)" 
                               placeholder="Enter custom Together model name" 
                               class="block w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                               value="${config.together_custom_model}">
                        <p class="text-xs text-gray-500 mt-1">Enter the exact model name as it appears in Together AI.</p>
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Together API Key</label>
                    <input type="password" onchange="updateModelConfig(${config.id}, 'together_api_key', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" placeholder="TOGETHER_API_KEY" value="${config.together_api_key}">
                    <p class="text-xs text-gray-500 mt-1">If empty, backend will use TOGETHER_API_KEY from environment.</p>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Logprobs (0-5)</label>
                    <input type="number" min="0" max="5" onchange="updateModelConfig(${config.id}, 'together_logprobs', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" value="${config.together_logprobs}">
                    <p class="text-xs text-gray-500 mt-1">If 0, logprobs will not be requested.</p>
                </div>
                ` : ''}
                
                ${config.inference_mode !== 'together_api' ? `
                <div>
                    <label class="inline-flex items-center mt-6">
                        <input type="checkbox" ${config.enable_prob_tracking ? 'checked' : ''} onchange="updateModelConfig(${config.id}, 'enable_prob_tracking', this.checked)" class="form-checkbox h-5 w-5 text-blue-600">
                        <span class="ml-2 text-sm text-gray-700">Enable probability tracking (requires vLLM)</span>
                    </label>
                    <p class="text-xs text-gray-500 mt-1">If enabled, vLLM will be used and probabilities will be recorded to a separate JSONL.</p>
                </div>
                <div>
                    <label class="inline-flex items-center mt-4">
                        <input type="checkbox" ${config.enable_path_vectors ? 'checked' : ''} onchange="updateModelConfig(${config.id}, 'enable_path_vectors', this.checked)" class="form-checkbox h-5 w-5 text-purple-600">
                        <span class="ml-2 text-sm text-gray-700">Enable path vectors (high memory usage)</span>
                    </label>
                    <p class="text-xs text-gray-500 mt-1">Records full probability distributions for Path of Distributions visualization. Requires probability tracking.</p>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Max Path Steps</label>
                    <input type="number" onchange="updateModelConfig(${config.id}, 'max_path_steps', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" value="${config.max_path_steps}" placeholder="0 for unlimited">
                    <p class="text-xs text-gray-500 mt-1">Maximum number of steps to record for path vectors. Use 0 or negative for unlimited (high memory usage).</p>
                </div>
                ` : ''}
                
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
            // Determine candidate models based on inference mode
            let models = [];
            if (config.inference_mode === 'together_api') {
                if (config.together_model === 'Custom Together Model') {
                    if (!config.together_custom_model || !config.together_custom_model.trim()) {
                        throw new Error(`Custom Together model name is required when 'Custom Together Model' is selected for model configuration ${config.id}`);
                    }
                    models = [config.together_custom_model.trim()];
                } else {
                    models = [config.together_model];
                }
            } else {
                models = (config.model === 'Link from Hugging Face' && config.customModel) ? [config.customModel] : [config.model];
            }
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

                                            // actual model already selected in models iteration
                                            const actualModel = model;

                                            const requestData = {
                                                model: actualModel,
                                                dataset: dataset,
                                                prompt: config.prompt_type === 'custom' ? config.prompt.trim() : '',
                                                prompt_type: config.prompt_type,
                                                backend: config.backend,
                                                use_together_api: config.inference_mode === 'together_api',
                                                together_api_key: config.inference_mode === 'together_api' ? (config.together_api_key || '') : '',
                                                together_logprobs: config.inference_mode === 'together_api' ? parseInt(config.together_logprobs || '0') : 0,
                                                temperature: parseFloat(temp),
                                                top_p: parseFloat(top_p),
                                                top_k: parseInt(top_k),
                                                seed: parseInt(seed),
                                                eval_method: config.eval_method,
                                                k: parseInt(k),
                                                max_tokens: parseInt(max_token),
                                                enable_prob_tracking: !!config.enable_prob_tracking,
                                                enable_path_vectors: !!config.enable_path_vectors,
                                                max_path_steps: parseInt(config.max_path_steps || '0')
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
                    <button onclick="openTruncationAnalysisForJob('${job.job_id}')" class="px-3 py-1 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700">📊 Truncation Analysis</button>
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

async function openProbPlotModal(jobId) {
    // First, get job info to check if dataset is math
    let isMathDataset = false;
    try {
        const jobResponse = await fetch(`${API_BASE}/jobs/${jobId}`);
        if (jobResponse.ok) {
            const jobInfo = await jobResponse.json();
            const dataset = jobInfo.request?.dataset || '';
            // Only treat "math" as a math dataset, not "sat_math"
            isMathDataset = dataset.toLowerCase() === 'math';
        }
    } catch (e) {
        console.warn('Could not fetch job info for dataset check:', e);
    }

    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';

    // Add math level plotting options if this is a math dataset
    const mathLevelOptions = isMathDataset ? `
        <option value="level_single">Math Level (Single Level)</option>
        <option value="level_aggregate">Math Level (All Levels Comparison)</option>
        <option value="starting_tokens_by_level">Starting Tokens by Level</option>
        <option value="ending_tokens_by_level">Ending Tokens by Level</option>
    ` : '';

    const mathLevelControlsHTML = isMathDataset ? `
        <div id="math-level-controls" class="hidden">
            <div id="level-single-controls" class="hidden">
                <label class="block text-sm font-medium text-gray-700">Select Level</label>
                <select id="math-level-select" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                    <option value="1">Level 1</option>
                    <option value="2">Level 2</option>
                    <option value="3">Level 3</option>
                    <option value="4">Level 4</option>
                    <option value="5">Level 5</option>
                </select>
            </div>
            <div id="level-aggregate-controls" class="hidden">
                <div class="text-sm text-gray-600 p-3 bg-blue-50 rounded-md">
                    <strong>Aggregate Mode:</strong> This will generate 3 separate plots showing all 5 difficulty levels:
                    <ul class="list-disc list-inside mt-2">
                        <li>Correct Token Probability vs Step</li>
                        <li>Chosen Token Probability vs Step</li>
                        <li>Entropy vs Step</li>
                    </ul>
                </div>
            </div>
        </div>
    ` : '';

    modal.innerHTML = `
        <div class="bg-white rounded-lg p-6 w-full max-w-xl">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-lg font-semibold">Plot Probabilities ${isMathDataset ? '(Math Dataset)' : ''}</h3>
                <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 text-xl">&times;</button>
            </div>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700">Plot Type</label>
                    <select id="prob-plot-type" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        <option value="aggregate" selected>Aggregate (All Samples)</option>
                        <option value="correct_aggregate">Aggregate (Correct Answers Only)</option>
                        <option value="incorrect_aggregate">Aggregate (Incorrect Answers Only)</option>
                        <option value="correct_vs_incorrect">Correct vs Incorrect Comparison</option>
                        <option value="single">Single Sample</option>
                        <option value="path_aggregate">Path of Distributions (Aggregate)</option>
                        <option value="path_single">Path of Distributions (Single Sample)</option>
                        ${mathLevelOptions}
                    </select>
                </div>
                <div id="prob-sample-id-field" class="hidden">
                    <label class="block text-sm font-medium text-gray-700">Sample ID (idx)</label>
                    <input id="prob-sample-id" type="number" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" placeholder="Enter sample idx">
                </div>
                ${mathLevelControlsHTML}
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
    const mathLevelControls = modal.querySelector('#math-level-controls');
    const levelSingleControls = modal.querySelector('#level-single-controls');
    const levelAggregateControls = modal.querySelector('#level-aggregate-controls');

    plotTypeSel.addEventListener('change', () => {
        // Hide all conditional fields first
        if (sampleField) sampleField.classList.add('hidden');
        if (mathLevelControls) mathLevelControls.classList.add('hidden');
        if (levelSingleControls) levelSingleControls.classList.add('hidden');
        if (levelAggregateControls) levelAggregateControls.classList.add('hidden');

        // Show appropriate fields based on plot type
        if (plotTypeSel.value === 'single' || plotTypeSel.value === 'path_single') {
            if (sampleField) sampleField.classList.remove('hidden');
        } else if (plotTypeSel.value === 'level_single') {
            if (mathLevelControls) mathLevelControls.classList.remove('hidden');
            if (levelSingleControls) levelSingleControls.classList.remove('hidden');
        } else if (plotTypeSel.value === 'level_aggregate') {
            if (mathLevelControls) mathLevelControls.classList.remove('hidden');
            if (levelAggregateControls) levelAggregateControls.classList.remove('hidden');
        }
    });
    modal.querySelector('#prob-plot-run').addEventListener('click', async () => {
        const plotType = plotTypeSel.value;
        const sampleIdVal = modal.querySelector('#prob-sample-id')?.value;
        const mathLevelVal = modal.querySelector('#math-level-select')?.value;
        const params = new URLSearchParams();
        params.append('plot_type', plotType);

        if (plotType === 'single' || plotType === 'path_single') {
            if (!sampleIdVal) {
                alert('Please enter a sample idx for single plot');
                return;
            }
            params.append('sample_id', String(parseInt(sampleIdVal)));
        } else if (plotType === 'level_single') {
            if (!mathLevelVal) {
                alert('Please select a math level');
                return;
            }
            params.append('math_level', mathLevelVal);
        } else if (plotType === 'level_aggregate') {
            // No additional parameters needed for level aggregate
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

            // Handle different response types
            if (plotType === 'level_aggregate') {
                // For level aggregate, provide download link for ZIP file
                const downloadUrl = URL.createObjectURL(blob);
                imgContainer.innerHTML = `
                    <div class="text-center p-4 bg-green-50 border border-green-200 rounded-lg">
                        <h4 class="font-semibold text-green-800 mb-2">✅ Level Aggregate Plots Generated!</h4>
                        <p class="text-sm text-green-700 mb-3">
                            3 plots have been generated showing all 5 difficulty levels:<br>
                            • Correct Token Probability vs Step<br>
                            • Chosen Token Probability vs Step<br>
                            • Entropy vs Step
                        </p>
                        <a href="${downloadUrl}" download="math_level_plots.zip" 
                           class="inline-block px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
                            📥 Download All Plots (ZIP)
                        </a>
                    </div>
                `;
            } else {
                // For regular plots, display the image
                const imgUrl = URL.createObjectURL(blob);
                imgContainer.innerHTML = `<img src="${imgUrl}" class="max-h-[60vh] rounded border"/>`;
            }
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
            
            // Auto-load cached results if available
            const cachedData = getCachedCoTData(this.value);
            if (cachedData) {
                console.log('Auto-loading cached CoT analysis for job:', this.value);
                currentCoTData = cachedData;
                showCoTResults(cachedData, true); // true indicates cached data
            } else {
                // Hide any previous results if no cache exists
                hideCoTResults();
                hideCoTError();
            }
        } else {
            analyzeBtn.disabled = true;
            exportExcelBtn.disabled = false; // Keep enabled if cached data exists
            exportJsonBtn.disabled = false;  // Keep enabled if cached data exists
            hideCoTResults();
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

    // Check if we have cached data first
    const cachedData = getCachedCoTData(jobId);
    
    if (cachedData) {
        console.log('Using cached CoT analysis data for job:', jobId);
        currentCoTData = cachedData;
        showCoTResults(cachedData, true); // true indicates cached data
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


        // Cache the data
        cacheCoTData(jobId, analysisData);

        // Hide loading and show results
        console.log('About to hide loading...');
        hideCoTLoading();
        console.log('Loading hidden, about to show results...');
        showCoTResults(analysisData, false); // false indicates fresh data
        console.log('showCoTResults call completed');

    } catch (error) {
        console.error('Error running CoT analysis:', error);
        hideCoTLoading();
        showCoTError(`Analysis failed: ${error.message}`);
    }
}

function showCoTResults(data, isCached = false) {
    try {
        console.log('CoT Analysis Data received:', data);
        console.log('Job summary:', data.job_summary);
        console.log('Per sample metrics length:', data.per_sample_metrics?.length);

        console.log('About to show results section...');
        document.getElementById('cot-analysis-results').classList.remove('hidden');
        console.log('Results section shown');

        // Populate summary statistics for new pillars schema
        console.log('About to call populatePillarsSummary...');
        populatePillarsSummary(data.summary, isCached);
        
        // Check if this is the new pillars analysis
        const isPillarsAnalysis = data.analysis_method === 'pillars_v2';
        if (isPillarsAnalysis) {
            console.log('Pillars v2 analysis detected, showing enhanced UI...');
            showPillarsAnalysisUI(data);
        } else {
            console.log('Legacy analysis detected, showing standard UI...');
            showLegacyAnalysisUI(data);
        }
        console.log('populatePillarsSummary completed');

        // Skip old CQS components - we're using comprehensive analysis now
        console.log('Skipping old CQS components - using comprehensive analysis');
        console.log('populateCQSComponents completed');

        // Enable export buttons
        const exportExcelBtn = document.getElementById('export-excel-btn');
        const exportJsonBtn = document.getElementById('export-json-btn');
        if (exportExcelBtn) exportExcelBtn.disabled = false;
        if (exportJsonBtn) exportJsonBtn.disabled = false;

        // Show random samples by default
        console.log('About to call showRandomSamples...');
        showRandomSamples();
        console.log('showRandomSamples completed');
    } catch (error) {
        console.error('ERROR in showCoTResults:', error);
        console.error('Error stack:', error.stack);
    }
}

function populatePillarsSummary(summary, isCached = false) {
    console.log('Populating Pillars Summary with:', summary);
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

    console.log('About to populate pillars summary with total_samples:', summary.total_samples);

    try {
        // Extract data from new pillars schema
        const totalSamples = summary.total_samples || 0;
        const avgOverall = summary.avg_overall || 0;
        const avgFaithfulness = summary.avg_faithfulness || 0;
        const avgUtility = summary.avg_utility || 0;
        const avgCoherence = summary.avg_coherence || 0;
        const avgFactuality = summary.avg_factuality || 0;
        const totalFlags = summary.total_flags || 0;
        const flagsByPillar = summary.flags_by_pillar || {};
        const judgeCallRate = summary.judge_call_rate || 0;
        const judgeBudgetUsed = summary.judge_budget_used || 0;
        const judgeBudgetTotal = summary.judge_budget_total || 0;
        
        // Determine quality level based on overall score
        let qualityLevel, qualityColor, qualityIcon;
        if (avgOverall >= 0.8) {
            qualityLevel = 'Excellent';
            qualityColor = 'text-green-600';
            qualityIcon = '⭐';
        } else if (avgOverall >= 0.6) {
            qualityLevel = 'Good';
            qualityColor = 'text-blue-600';
            qualityIcon = '👍';
        } else if (avgOverall >= 0.4) {
            qualityLevel = 'Fair';
            qualityColor = 'text-yellow-600';
            qualityIcon = '⚠️';
        } else {
            qualityLevel = 'Poor';
            qualityColor = 'text-red-600';
            qualityIcon = '❌';
        }

        // Add cache indicator
        const cacheIndicator = isCached ? 
            '<div class="mb-2 text-sm text-amber-600 bg-amber-50 px-2 py-1 rounded border border-amber-200">💾 Cached Data (click "Run CoT Analysis" to refresh)</div>' : 
            '<div class="mb-2 text-sm text-green-600 bg-green-50 px-2 py-1 rounded border border-green-200">🔄 Fresh Analysis</div>';

        summaryDiv.innerHTML = `
            <div class="bg-white p-6 rounded-lg border shadow-sm">
                ${cacheIndicator}
                
                <!-- Header Section -->
                <div class="text-center mb-6">
                    <h4 class="text-xl font-bold text-gray-800 mb-2">🏛️ Four-Pillar Analysis</h4>
                    <p class="text-sm text-gray-600">Faithfulness + Utility + Coherence + Factuality</p>
            </div>

                <!-- Main Layout: Overall Score + Four Pillars -->
                <div class="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8 items-stretch">
                    <!-- Overall Score (Left Side) -->
                    <div class="lg:col-span-1 flex">
                        <div class="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl border-2 border-blue-200 w-full flex flex-col justify-center">
                            <div class="text-4xl font-bold text-blue-700 mb-2">${avgOverall.toFixed(3)}</div>
                            <div class="text-lg font-medium text-blue-600 mb-2">Overall Score</div>
                            <div class="text-sm text-blue-500">${qualityIcon} ${qualityLevel}</div>
            </div>
            </div>
                    
                    <!-- Four Pillars (Right Side - Takes Full Width) -->
                    <div class="lg:col-span-4 flex flex-col">
                        <h5 class="text-lg font-semibold text-gray-700 mb-4 text-center">Four Pillars Breakdown</h5>
                        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 flex-1">
                            <!-- Faithfulness -->
                            <div class="text-center p-4 bg-red-50 rounded-xl border-2 border-red-200 hover:shadow-md transition-shadow flex flex-col justify-center min-h-[120px]">
                                <div class="text-2xl font-bold text-red-600 mb-2">${avgFaithfulness.toFixed(3)}</div>
                                <div class="text-sm font-medium text-red-700 mb-2">Faithfulness</div>
                                <div class="text-xs text-red-500 bg-red-100 px-2 py-1 rounded-full inline-block">${flagsByPillar.faithfulness || 0} flags</div>
            </div>
                            
                            <!-- Utility -->
                            <div class="text-center p-4 bg-green-50 rounded-xl border-2 border-green-200 hover:shadow-md transition-shadow flex flex-col justify-center min-h-[120px]">
                                <div class="text-2xl font-bold text-green-600 mb-2">${avgUtility.toFixed(3)}</div>
                                <div class="text-sm font-medium text-green-700 mb-2">Utility</div>
                                <div class="text-xs text-green-500 bg-green-100 px-2 py-1 rounded-full inline-block">${flagsByPillar.utility || 0} flags</div>
            </div>
                            
                            <!-- Coherence -->
                            <div class="text-center p-4 bg-purple-50 rounded-xl border-2 border-purple-200 hover:shadow-md transition-shadow flex flex-col justify-center min-h-[120px]">
                                <div class="text-2xl font-bold text-purple-600 mb-2">${avgCoherence.toFixed(3)}</div>
                                <div class="text-sm font-medium text-purple-700 mb-2">Coherence</div>
                                <div class="text-xs text-purple-500 bg-purple-100 px-2 py-1 rounded-full inline-block">${flagsByPillar.coherence || 0} flags</div>
            </div>
                            
                            <!-- Factuality -->
                            <div class="text-center p-4 bg-orange-50 rounded-xl border-2 border-orange-200 hover:shadow-md transition-shadow flex flex-col justify-center min-h-[120px]">
                                <div class="text-2xl font-bold text-orange-600 mb-2">${avgFactuality.toFixed(3)}</div>
                                <div class="text-sm font-medium text-orange-700 mb-2">Factuality</div>
                                <div class="text-xs text-orange-500 bg-orange-100 px-2 py-1 rounded-full inline-block">${flagsByPillar.factuality || 0} flags</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Statistics Section -->
                <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                    <div class="text-center p-4 bg-gray-50 rounded-lg border border-gray-200 flex flex-col justify-center min-h-[80px]">
                        <div class="text-lg font-bold text-gray-700">${totalSamples.toLocaleString()}</div>
                        <div class="text-sm text-gray-600">Total Samples</div>
                    </div>
                    <div class="text-center p-4 bg-red-50 rounded-lg border border-red-200 flex flex-col justify-center min-h-[80px]">
                        <div class="text-lg font-bold text-red-600">${totalFlags.toLocaleString()}</div>
                        <div class="text-sm text-red-600">Total Flags</div>
                    </div>
                    <div class="text-center p-4 bg-indigo-50 rounded-lg border border-indigo-200 flex flex-col justify-center min-h-[80px]">
                        <div class="text-lg font-bold text-indigo-600">${(judgeCallRate * 100).toFixed(1)}%</div>
                        <div class="text-sm text-indigo-600">Judge Call Rate</div>
                    </div>
                    <div class="text-center p-4 bg-blue-50 rounded-lg border border-blue-200 flex flex-col justify-center min-h-[80px]">
                        <div class="text-lg font-bold text-blue-600">${judgeBudgetUsed}/${judgeBudgetTotal.toLocaleString()}</div>
                        <div class="text-sm text-blue-600">Judge Budget Used</div>
                    </div>
                </div>

                <!-- Performance Insights -->
                <div class="p-6 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border border-gray-200">
                    <h5 class="font-semibold text-gray-700 mb-4 text-center">📊 Performance Insights</h5>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                        <div>
                            <div class="font-medium text-gray-600">Flag Distribution:</div>
                            <div class="mt-1 space-y-1">
                                <div class="flex justify-between">
                                    <span class="text-red-600">Faithfulness:</span>
                                    <span class="font-medium">${flagsByPillar.faithfulness || 0} (${((flagsByPillar.faithfulness || 0) / Math.max(totalFlags, 1) * 100).toFixed(1)}%)</span>
                </div>
                                <div class="flex justify-between">
                                    <span class="text-green-600">Utility:</span>
                                    <span class="font-medium">${flagsByPillar.utility || 0} (${((flagsByPillar.utility || 0) / Math.max(totalFlags, 1) * 100).toFixed(1)}%)</span>
            </div>
                                <div class="flex justify-between">
                                    <span class="text-purple-600">Coherence:</span>
                                    <span class="font-medium">${flagsByPillar.coherence || 0} (${((flagsByPillar.coherence || 0) / Math.max(totalFlags, 1) * 100).toFixed(1)}%)</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-orange-600">Factuality:</span>
                                    <span class="font-medium">${flagsByPillar.factuality || 0} (${((flagsByPillar.factuality || 0) / Math.max(totalFlags, 1) * 100).toFixed(1)}%)</span>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div class="font-medium text-gray-600">Analysis Status:</div>
                            <div class="mt-1 space-y-1">
                                <div class="flex justify-between">
                                    <span>Analysis Method:</span>
                                    <span class="font-medium text-blue-600">Four-Pillar v2</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Judge Integration:</span>
                                    <span class="font-medium ${judgeCallRate > 0 ? 'text-green-600' : 'text-gray-600'}">${judgeCallRate > 0 ? 'Active' : 'Standby'}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Data Status:</span>
                                    <span class="font-medium ${isCached ? 'text-amber-600' : 'text-green-600'}">${isCached ? 'Cached' : 'Fresh'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        console.log('Pillars Summary populated successfully');
    } catch (error) {
        console.error('Error populating pillars summary:', error);
        summaryDiv.innerHTML = '<div class="text-red-500">Error populating summary</div>';
    }
}

// CQS Components function removed - using comprehensive analysis instead

function showTopPerformers() {
    if (!currentCoTData) return;

    const samples = currentCoTData.per_sample
        .filter(s => s.scores && s.scores.overall !== undefined)
        .sort((a, b) => (b.scores.overall || 0) - (a.scores.overall || 0))
        .slice(0, 10);

    renderSampleAnalysis(samples, 'Top 10 Performers (Highest Overall Scores)');
}

function showBottomPerformers() {
    if (!currentCoTData) return;

    const samples = currentCoTData.per_sample
        .filter(s => s.scores && s.scores.overall !== undefined)
        .sort((a, b) => (a.scores.overall || 0) - (b.scores.overall || 0))
        .slice(0, 10);

    renderSampleAnalysis(samples, 'Bottom 10 Performers (Lowest Overall Scores)');
}

function showRandomSamples() {
    console.log('ShowRandomSamples called, currentCoTData:', currentCoTData);

    if (!currentCoTData) {
        console.error('No currentCoTData available for samples');
        return;
    }

    if (!currentCoTData.per_sample) {
        console.error('No per_sample in currentCoTData');
        return;
    }

    console.log('Available samples:', currentCoTData.per_sample.length);

    const samples = [...currentCoTData.per_sample]
        .filter(s => s.scores && s.scores.overall !== undefined)
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
            ${samples.map((sample, index) => {
                const score = sample.scores?.overall || 0;
                const scoreColor = getScoreColor(score);
                const correctIcon = sample.evidence?.final_correct ? '✅' : '❌';
                const flagsCount = sample.flags?.length || 0;
                
                return `
                <div class="border border-gray-200 rounded-lg p-3 bg-white">
                    <div class="flex justify-between items-start mb-2">
                            <span class="text-sm font-medium text-gray-700">Sample #${index + 1}</span>
                            <span class="text-sm font-bold ${scoreColor}">
                                Overall: ${score.toFixed(3)}
                        </span>
                    </div>
                    
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs mb-2">
                            <div class="text-center p-1 bg-red-50 rounded">
                                <div class="font-bold text-red-600">${(sample.scores?.faithfulness || 0).toFixed(2)}</div>
                                <div class="text-red-700">Faith</div>
                            </div>
                            <div class="text-center p-1 bg-green-50 rounded">
                                <div class="font-bold text-green-600">${(sample.scores?.utility || 0).toFixed(2)}</div>
                                <div class="text-green-700">Utility</div>
                            </div>
                            <div class="text-center p-1 bg-purple-50 rounded">
                                <div class="font-bold text-purple-600">${(sample.scores?.coherence || 0).toFixed(2)}</div>
                                <div class="text-purple-700">Coherence</div>
                            </div>
                            <div class="text-center p-1 bg-orange-50 rounded">
                                <div class="font-bold text-orange-600">${(sample.scores?.factuality || 0).toFixed(2)}</div>
                                <div class="text-orange-700">Factuality</div>
                            </div>
                    </div>
                    
                    <div class="text-xs text-gray-600">
                            Flags: ${flagsCount} | 
                            Arith Errors: ${sample.evidence?.arith_bad_examples?.length || 0} | 
                            Correct: ${correctIcon}
                    </div>
                        
                        ${sample.flags && sample.flags.length > 0 ? `
                            <div class="mt-2">
                                <div class="text-xs text-red-600">⚠️ ${sample.flags.slice(0, 2).map(f => `${f.pillar}: ${f.issue}`).join(', ')}${sample.flags.length > 2 ? '...' : ''}</div>
                </div>
                        ` : ''}
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

// Helper function for score color coding
function getScoreColor(score) {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-blue-600';
    if (score >= 0.4) return 'text-yellow-600';
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

function openTruncationAnalysisForJob(jobId) {
    try {
        console.log(`Opening Truncation analysis for job: ${jobId}`);

        // Switch to Truncation Analysis tab
        showTab('truncation-analysis');

        // Load jobs for the dropdown first, then wait for elements
        loadTruncationJobs().then(() => {
            // Wait for the truncation-job-select element to be available
            waitForElement('#truncation-job-select').then((jobSelect) => {
                // Select the job
                jobSelect.value = jobId;
                console.log(`Selected job ${jobId} in dropdown`);

                // Enable the analyze button
                const analyzeBtn = document.getElementById('truncation-analyze-btn');
                if (analyzeBtn) {
                    analyzeBtn.disabled = false;
                    console.log('Analyze button enabled');
                }

                // Optionally run analysis immediately
                setTimeout(() => {
                    if (confirm('Run Truncation analysis for this job now?')) {
                        runTruncationAnalysis();
                    }
                }, 100);
            }).catch(error => {
                console.error('Error waiting for truncation-job-select element:', error);
                alert('Truncation Analysis tab not properly loaded. Please try refreshing the page.');
            });
        }).catch(error => {
            console.error('Error loading Truncation jobs:', error);
            alert('Error loading jobs for Truncation analysis: ' + error.message);
        });
    } catch (error) {
        console.error('Error in openTruncationAnalysisForJob:', error);
        alert('Error opening Truncation analysis: ' + error.message);
    }
}

async function exportCoTExcel() {
    if (!currentCoTData) {
        showCoTError('Please run CoT analysis first');
        return;
    }

    try {
        // Create a comprehensive Excel export with CoT analysis data
        await exportCoTAnalysisToExcel(currentCoTData);
    } catch (error) {
        console.error('Error exporting CoT analysis to Excel:', error);
        
        // Try fallback CSV export
        try {
            console.log('Attempting fallback CSV export...');
            await exportCoTAnalysisToCSV(currentCoTData);
        } catch (csvError) {
            console.error('CSV export also failed:', csvError);
            showCoTError(`Excel export failed: ${error.message}. CSV export also failed.`);
        }
    }
}

// Fallback CSV export function
async function exportCoTAnalysisToCSV(cotData) {
    try {
        console.log('Creating CSV export...');
        
        // Create CSV content
        let csvContent = 'CoT Analysis Export\n\n';
        
        // Summary section
        csvContent += 'SUMMARY\n';
        csvContent += `Job ID,${cotData.job_id || 'N/A'}\n`;
        csvContent += `Analysis Method,${cotData.analysis_method || 'N/A'}\n`;
        csvContent += `Total Samples,${cotData.summary?.total_samples || 0}\n`;
        csvContent += `Overall Score,${cotData.summary?.avg_overall || 0}\n`;
        csvContent += `Faithfulness,${cotData.summary?.avg_faithfulness || 0}\n`;
        csvContent += `Utility,${cotData.summary?.avg_utility || 0}\n`;
        csvContent += `Coherence,${cotData.summary?.avg_coherence || 0}\n`;
        csvContent += `Factuality,${cotData.summary?.avg_factuality || 0}\n\n`;
        
        // Detailed data
        csvContent += 'DETAILED ANALYSIS\n';
        csvContent += 'Sample #,Problem,Model Output,Final Answer Correct,Overall Score,Faithfulness,Utility,Coherence,Factuality,Flag Count,Flags\n';
        
        if (cotData.per_sample && Array.isArray(cotData.per_sample)) {
            cotData.per_sample.forEach((sample, index) => {
                const flags = sample.flags || [];
                const flagDescriptions = flags.map(f => `${f.pillar || 'Unknown'}: ${f.issue || 'Unknown'}`).join('; ');
                
                csvContent += `${index + 1},"${(sample.problem || 'N/A').replace(/"/g, '""')}","${(sample.model_output || 'N/A').replace(/"/g, '""')}",${sample.evidence?.final_correct ? 'Yes' : 'No'},${sample.scores?.overall || 0},${sample.scores?.faithfulness || 0},${sample.scores?.utility || 0},${sample.scores?.coherence || 0},${sample.scores?.factuality || 0},${flags.length},"${flagDescriptions.replace(/"/g, '""')}"\n`;
            });
        }
        
        // Create and download CSV file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `cot_analysis_${cotData.job_id || 'unknown'}_${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
        
        console.log('CSV export completed successfully');
        
    } catch (error) {
        console.error('Error creating CSV export:', error);
        throw error;
    }
}

async function exportCoTAnalysisToExcel(cotData) {
    try {
        console.log('Starting Excel export...', cotData);
        
        // Import the required libraries for Excel generation
        let XLSX;
        try {
            XLSX = await import('https://cdn.sheetjs.com/xlsx-0.20.1/package/xlsx.mjs');
            console.log('XLSX library loaded successfully');
        } catch (importError) {
            console.error('Failed to import XLSX library:', importError);
            throw new Error('Failed to load Excel export library. Please check your internet connection.');
        }
        
        // Validate data structure
        if (!cotData || !cotData.summary) {
            throw new Error('Invalid data structure: missing summary data');
        }
        
        if (!cotData.per_sample || !Array.isArray(cotData.per_sample)) {
            throw new Error('Invalid data structure: missing per_sample data');
        }
        
        console.log('Data validation passed, creating workbook...');
        
        // Prepare the data for Excel export
        const workbook = XLSX.utils.book_new();
        
        // 1. Summary Sheet
        const summaryData = [
            ['CoT Analysis Summary'],
            [''],
            ['Job ID', cotData.job_id || 'N/A'],
            ['Analysis Method', cotData.analysis_method || 'N/A'],
            ['Timestamp', cotData.timestamp || new Date().toISOString()],
            ['Total Samples', cotData.summary.total_samples || 0],
            [''],
            ['Overall Scores'],
            ['Overall Score', cotData.summary.avg_overall || 0],
            ['Faithfulness', cotData.summary.avg_faithfulness || 0],
            ['Utility', cotData.summary.avg_utility || 0],
            ['Coherence', cotData.summary.avg_coherence || 0],
            ['Factuality', cotData.summary.avg_factuality || 0],
            [''],
            ['Flag Statistics'],
            ['Total Flags', cotData.summary.total_flags || 0],
            ['Faithfulness Flags', cotData.summary.flags_by_pillar?.faithfulness || 0],
            ['Utility Flags', cotData.summary.flags_by_pillar?.utility || 0],
            ['Coherence Flags', cotData.summary.flags_by_pillar?.coherence || 0],
            ['Factuality Flags', cotData.summary.flags_by_pillar?.factuality || 0],
            [''],
            ['Judge Statistics'],
            ['Judge Call Rate', `${((cotData.summary.judge_call_rate || 0) * 100).toFixed(1)}%`],
            ['Budget Used', `${cotData.summary.judge_budget_used || 0}/${cotData.summary.judge_budget_total || 0}`]
        ];
        
        const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
        XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary');
        
        // 2. Detailed Analysis Sheet
        const detailedData = [
            [
                'Sample #', 'Problem', 'Model Output', 'Final Answer Correct',
                'Overall Score', 'Faithfulness Score', 'Utility Score', 'Coherence Score', 'Factuality Score',
                'Flag Count', 'Flags', 'Evidence Summary', 'Judge Scores', 'Arithmetic Errors'
            ]
        ];
        
        cotData.per_sample.forEach((sample, index) => {
            try {
                const flags = sample.flags || [];
                const flagDescriptions = flags.map(f => `${f.pillar || 'Unknown'}: ${f.issue || 'Unknown issue'}`).join('; ');
                const arithErrors = sample.evidence?.arith_bad_examples?.length || 0;
                const judgeScores = sample.judge_raw ? JSON.stringify(sample.judge_raw) : 'N/A';
                
                detailedData.push([
                    index + 1,
                    (sample.problem || 'N/A').substring(0, 1000), // Limit length
                    (sample.model_output || 'N/A').substring(0, 2000), // Limit length
                    sample.evidence?.final_correct ? 'Yes' : 'No',
                    sample.scores?.overall || 0,
                    sample.scores?.faithfulness || 0,
                    sample.scores?.utility || 0,
                    sample.scores?.coherence || 0,
                    sample.scores?.factuality || 0,
                    flags.length,
                    flagDescriptions || 'None',
                    `Final: ${sample.evidence?.final_correct ? 'Correct' : 'Incorrect'}, Intermediate OK: ${(sample.evidence?.intermediate_ok_rate || 0).toFixed(2)}`,
                    judgeScores,
                    arithErrors
                ]);
            } catch (sampleError) {
                console.warn(`Error processing sample ${index}:`, sampleError);
                // Add a row with error info
                detailedData.push([
                    index + 1,
                    'ERROR',
                    'Failed to process sample',
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                    0, 'Error processing', 'N/A', 'N/A', 0
                ]);
            }
        });
        
        const detailedSheet = XLSX.utils.aoa_to_sheet(detailedData);
        
        // Set column widths for better readability
        detailedSheet['!cols'] = [
            { wch: 8 },  // Sample #
            { wch: 30 }, // Problem
            { wch: 50 }, // Model Output
            { wch: 15 }, // Final Answer Correct
            { wch: 12 }, // Overall Score
            { wch: 15 }, // Faithfulness Score
            { wch: 12 }, // Utility Score
            { wch: 15 }, // Coherence Score
            { wch: 15 }, // Factuality Score
            { wch: 10 }, // Flag Count
            { wch: 40 }, // Flags
            { wch: 50 }, // Evidence Summary
            { wch: 30 }, // Judge Scores
            { wch: 15 }  // Arithmetic Errors
        ];
        
        XLSX.utils.book_append_sheet(workbook, detailedSheet, 'Detailed Analysis');
        
        // 3. Flags Summary Sheet
        const flagData = [['Pillar', 'Issue Type', 'Count']];
        const flagCounts = {};
        
        cotData.per_sample.forEach(sample => {
            (sample.flags || []).forEach(flag => {
                const key = `${flag.pillar}: ${flag.issue}`;
                flagCounts[key] = (flagCounts[key] || 0) + 1;
            });
        });
        
        Object.entries(flagCounts).forEach(([flag, count]) => {
            const [pillar, issue] = flag.split(': ');
            flagData.push([pillar, issue, count]);
        });
        
        const flagSheet = XLSX.utils.aoa_to_sheet(flagData);
        XLSX.utils.book_append_sheet(workbook, flagSheet, 'Flag Summary');
        
        // 4. Raw Data Sheet (for advanced users)
        const rawData = [['Raw JSON Data']];
        rawData.push([JSON.stringify(cotData, null, 2)]);
        
        const rawSheet = XLSX.utils.aoa_to_sheet(rawData);
        XLSX.utils.book_append_sheet(workbook, rawSheet, 'Raw Data');
        
        // Generate and download the Excel file
        console.log('Generating Excel file...');
        const fileName = `cot_analysis_${cotData.job_id || 'unknown'}_${new Date().toISOString().split('T')[0]}.xlsx`;
        
        try {
            XLSX.writeFile(workbook, fileName);
            console.log('CoT analysis exported to Excel successfully');
        } catch (writeError) {
            console.error('Error writing Excel file:', writeError);
            throw new Error('Failed to write Excel file. This might be due to browser security restrictions.');
        }
        
    } catch (error) {
        console.error('Error creating CoT Excel export:', error);
        throw error;
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
    const resultsDiv = document.getElementById('cot-analysis-results');
    if (resultsDiv) {
        resultsDiv.classList.add('hidden');
        // Clear the results content to prevent stale data from showing
        const summaryDiv = document.getElementById('cot-summary');
        const samplesDiv = document.getElementById('sample-analysis');
        if (summaryDiv) summaryDiv.innerHTML = '';
        if (samplesDiv) samplesDiv.innerHTML = '';
    }
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

// OpenAI API Key functions
function toggleOpenAIKeyVisibility() {
    const keyInput = document.getElementById('openai_api_key');
    if (keyInput.type === 'password') {
        keyInput.type = 'text';
    } else {
        keyInput.type = 'password';
    }
}

function toggleHFTokenVisibility() {
    const keyInput = document.getElementById('hf_token');
    if (keyInput.type === 'password') {
        keyInput.type = 'text';
    } else {
        keyInput.type = 'password';
    }
}

async function saveConfiguration() {
    try {
        const apiKey = document.getElementById('openai_api_key').value;
        
        // Save the configuration
        const config = {
            openai_api_key: apiKey
        };
        
        const response = await fetch(`${API_BASE}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            alert('✅ Configuration saved successfully!');
        } else {
            const errorData = await response.json();
            alert(`❌ Failed to save configuration: ${errorData.detail || 'Unknown error'}`);
        }

    } catch (error) {
        console.error('Error saving configuration:', error);
        alert('❌ Error saving configuration: ' + error.message);
    }
}

async function testOpenAIKey() {
    const apiKey = document.getElementById('openai_api_key').value;
    if (!apiKey) {
        alert('Please enter an OpenAI API key first');
        return;
    }

    try {
        // Show loading state
        const testButton = event.target;
        const originalText = testButton.textContent;
        testButton.textContent = 'Testing...';
        testButton.disabled = true;

        // Test the API key by calling the backend
        const response = await fetch(`${API_BASE}/config/test-openai-key`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ api_key: apiKey })
        });

        const data = await response.json();
        
        if (response.ok && data.valid) {
            alert('✅ OpenAI API key is valid and working!');
        } else {
            alert(`❌ OpenAI API key test failed: ${data.error || 'Unknown error'}`);
        }

        // Restore button state
        testButton.textContent = originalText;
        testButton.disabled = false;

    } catch (error) {
        console.error('Error testing OpenAI API key:', error);
        alert('❌ Error testing API key: ' + error.message);
        
        // Restore button state
        const testButton = event.target;
        testButton.textContent = 'Test API Key';
        testButton.disabled = false;
    }
}

// Truncation Analysis Functions
let currentTruncationAnalysis = null;

async function loadTruncationJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();

        const select = document.getElementById('truncation-job-select');
        select.innerHTML = '<option value="">Select a completed job...</option>';

        data.jobs.forEach(job => {
            if (job.status === 'DONE' && job.result_file) {
                const option = document.createElement('option');
                option.value = job.job_id;
                option.textContent = `${job.job_id} - ${job.model || 'Unknown Model'} - ${job.dataset || 'Unknown Dataset'}`;
                select.appendChild(option);
            }
        });

        console.log(`Loaded ${data.jobs.filter(j => j.status === 'DONE').length} completed jobs for truncation analysis`);
    } catch (error) {
        console.error('Error loading jobs for truncation analysis:', error);
        showTruncationError('Failed to load jobs: ' + error.message);
    }
}

function validateTruncationInputs() {
    const jobId = document.getElementById('truncation-job-select').value;
    const modelPath = document.getElementById('truncation-model-path').value.trim();
    const datasetName = document.getElementById('truncation-dataset-name').value.trim();
    const backend = document.getElementById('truncation-backend').value;
    const temperature = parseFloat(document.getElementById('truncation-temperature').value);
    const topP = parseFloat(document.getElementById('truncation-top-p').value);

    const isValid = jobId && modelPath && datasetName && backend &&
        !isNaN(temperature) && temperature >= 0 && temperature <= 2 &&
        !isNaN(topP) && topP >= 0 && topP <= 1;

    const analyzeBtn = document.getElementById('truncation-analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = !isValid;
    }

    // Clear any previous error messages when inputs are valid
    if (isValid) {
        const errorDiv = document.getElementById('truncation-error');
        if (errorDiv) {
            errorDiv.classList.add('hidden');
        }
    }

    return isValid;
}

async function runTruncationAnalysis() {
    if (!validateTruncationInputs()) {
        return;
    }

    const jobId = document.getElementById('truncation-job-select').value;
    const modelPath = document.getElementById('truncation-model-path').value.trim();
    const datasetName = document.getElementById('truncation-dataset-name').value.trim();
    const backend = document.getElementById('truncation-backend').value;
    const temperature = parseFloat(document.getElementById('truncation-temperature').value);
    const topP = parseFloat(document.getElementById('truncation-top-p').value);

    // Show loading state
    document.getElementById('truncation-loading').classList.remove('hidden');
    document.getElementById('truncation-analysis-results').classList.add('hidden');
    document.getElementById('truncation-error').classList.add('hidden');
    document.getElementById('truncation-analyze-btn').disabled = true;

    try {
        const request = {
            job_id: jobId,
            model_name_or_path: modelPath,
            dataset_name: datasetName,
            backend: backend,
            temperature: temperature,
            top_p: topP
        };

        console.log('Starting truncation analysis:', request);

        const response = await fetch(`${API_BASE}/jobs/${jobId}/truncation-analysis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const result = await response.json();
        currentTruncationAnalysis = result;

        console.log('Truncation analysis completed:', result);
        displayTruncationResults(result);

    } catch (error) {
        console.error('Error running truncation analysis:', error);
        showTruncationError('Analysis failed: ' + error.message);
    } finally {
        document.getElementById('truncation-loading').classList.add('hidden');
        document.getElementById('truncation-analyze-btn').disabled = false;
    }
}

// Comprehensive Analysis UI Functions
function showPillarsAnalysisUI(data) {
    console.log('Showing pillars analysis UI...');
    
    // The summary is already populated by populatePillarsSummary
    // This function can be used for additional pillars-specific UI elements
    
    // Update any additional UI elements specific to pillars analysis
    console.log('Pillars analysis UI displayed successfully');
}

function showLegacyAnalysisUI(data) {
    console.log('Showing legacy analysis UI...');
    // Keep existing legacy UI behavior - call the existing function
    populateDetailedResults(data.per_sample_metrics);
}

// Legacy analysis UI function (simplified since detailed results section removed)
function populateDetailedResults(samples) {
    console.log('Legacy detailed results requested but section removed');
    return;
}

function displayTruncationResults(result) {
    const resultsDiv = document.getElementById('truncation-analysis-results');
    const summaryDiv = document.getElementById('truncation-summary');

    // Display summary
    summaryDiv.innerHTML = `
        <div class="bg-white p-4 rounded-lg border">
            <h4 class="font-semibold text-gray-900">Analysis Status</h4>
            <p class="text-sm text-gray-600">${result.status}</p>
        </div>
        <div class="bg-white p-4 rounded-lg border">
            <h4 class="font-semibold text-gray-900">Computation Time</h4>
            <p class="text-sm text-gray-600">${result.computation_time ? result.computation_time.toFixed(2) + 's' : 'N/A'}</p>
        </div>
        <div class="bg-white p-4 rounded-lg border">
            <h4 class="font-semibold text-gray-900">Raw Data</h4>
            <p class="text-sm text-gray-600">${result.raw_curves_path ? 'Available' : 'Not available'}</p>
        </div>
        <div class="bg-white p-4 rounded-lg border">
            <h4 class="font-semibold text-gray-900">Plots Generated</h4>
            <p class="text-sm text-gray-600">${result.correct_plot_path ? 'Correct ✓' : 'Correct ✗'} | ${result.incorrect_plot_path ? 'Incorrect ✓' : 'Incorrect ✗'}</p>
        </div>
    `;

    // Enable plot buttons if plots are available
    const correctBtn = document.getElementById('show-correct-plot-btn');
    const incorrectBtn = document.getElementById('show-incorrect-plot-btn');
    const downloadBtn = document.getElementById('download-truncation-btn');

    correctBtn.disabled = !result.correct_plot_path;
    incorrectBtn.disabled = !result.incorrect_plot_path;
    downloadBtn.disabled = !result.raw_curves_path;

    resultsDiv.classList.remove('hidden');
}

function showTruncationPlot(plotType) {
    if (!currentTruncationAnalysis) {
        showTruncationError('No analysis results available');
        return;
    }

    const jobId = currentTruncationAnalysis.job_id;
    const plotContainer = document.getElementById('truncation-plot-container');

    // Show loading state
    plotContainer.innerHTML = '<div class="text-center py-4"><div class="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div><p class="mt-2 text-gray-600">Loading plot...</p></div>';

    // Create image element
    const img = document.createElement('img');
    img.src = `${API_BASE}/jobs/${jobId}/truncation-analysis/plot?plot_type=${plotType}`;
    img.alt = `${plotType} samples confidence curve`;
    img.className = 'max-w-full h-auto mx-auto rounded-lg shadow-lg';

    img.onload = () => {
        plotContainer.innerHTML = '';
        plotContainer.appendChild(img);
    };

    img.onerror = () => {
        plotContainer.innerHTML = '<p class="text-red-600">Failed to load plot. Please try again.</p>';
    };
}

function downloadTruncationData() {
    if (!currentTruncationAnalysis || !currentTruncationAnalysis.raw_curves_path) {
        showTruncationError('No raw data available for download');
        return;
    }

    // Create a temporary link to download the file
    const link = document.createElement('a');
    link.href = `${API_BASE}/file?path=${encodeURIComponent(currentTruncationAnalysis.raw_curves_path)}`;
    link.download = `truncation_curves_${currentTruncationAnalysis.job_id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function showTruncationError(message) {
    const errorDiv = document.getElementById('truncation-error');
    const errorMessage = document.getElementById('truncation-error-message');

    errorMessage.textContent = message;
    errorDiv.classList.remove('hidden');

    // Hide other sections
    document.getElementById('truncation-analysis-results').classList.add('hidden');
    document.getElementById('truncation-loading').classList.add('hidden');
}

// Ensure DOM is loaded before running any functions
document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM loaded, initializing UI...');

    // Verify that all expected tab elements exist
    const expectedTabs = ['configure', 'monitor', 'jobs', 'cot-analysis', 'truncation-analysis', 'settings'];
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
    
    // Load cached data if available
    loadCachedCoTDataOnStartup();
});

// CoT Analysis Caching Functions
function cacheCoTData(jobId, data) {
    try {
        const cacheKey = `cot_analysis_${jobId}`;
        const cacheData = {
            data: data,
            timestamp: Date.now(),
            jobId: jobId
        };
        localStorage.setItem(cacheKey, JSON.stringify(cacheData));
        console.log(`Cached CoT analysis data for job ${jobId}`);
    } catch (error) {
        console.error('Error caching CoT analysis data:', error);
    }
}

function getCachedCoTData(jobId) {
    try {
        const cacheKey = `cot_analysis_${jobId}`;
        const cached = localStorage.getItem(cacheKey);
        
        if (!cached) {
            return null;
        }
        
        const cacheData = JSON.parse(cached);
        
        // Check if cache is still valid (24 hours)
        const maxAge = 24 * 60 * 60 * 1000; // 24 hours in milliseconds
        const age = Date.now() - cacheData.timestamp;
        
        if (age > maxAge) {
            console.log(`Cache expired for job ${jobId}, removing...`);
            localStorage.removeItem(cacheKey);
            return null;
        }
        
        console.log(`Found cached data for job ${jobId} (age: ${Math.round(age / 60000)} minutes)`);
        return cacheData.data;
    } catch (error) {
        console.error('Error retrieving cached CoT analysis data:', error);
        return null;
    }
}

function clearCoTCache(jobId = null) {
    try {
        let clearedCount = 0;
        
        if (jobId) {
            const cacheKey = `cot_analysis_${jobId}`;
            localStorage.removeItem(cacheKey);
            clearedCount = 1;
            console.log(`Cleared cache for job ${jobId}`);
        } else {
            // Clear all CoT analysis cache
            const keys = Object.keys(localStorage);
            keys.forEach(key => {
                if (key.startsWith('cot_analysis_')) {
                    localStorage.removeItem(key);
                    clearedCount++;
                }
            });
            console.log(`Cleared ${clearedCount} cached CoT analyses`);
        }
        
        // Show feedback to user
        if (clearedCount > 0) {
            showCoTCacheFeedback(`Cleared ${clearedCount} cached analysis${clearedCount > 1 ? 'es' : ''}`);
        } else {
            showCoTCacheFeedback('No cached analyses found');
        }
        
        // Clear current data if it was cached
        if (currentCoTData && jobId && currentCoTData.job_id === jobId) {
            currentCoTData = null;
            hideCoTResults();
        }
        
    } catch (error) {
        console.error('Error clearing CoT cache:', error);
        showCoTCacheFeedback('Error clearing cache');
    }
}

function showCoTCacheFeedback(message) {
    // Create or update feedback element
    let feedbackEl = document.getElementById('cache-feedback');
    if (!feedbackEl) {
        feedbackEl = document.createElement('div');
        feedbackEl.id = 'cache-feedback';
        feedbackEl.className = 'fixed top-4 right-4 z-50 px-4 py-2 rounded-md shadow-lg';
        document.body.appendChild(feedbackEl);
    }
    
    feedbackEl.innerHTML = `
        <div class="bg-blue-100 border border-blue-300 text-blue-800 px-3 py-2 rounded-md">
            ${message}
        </div>
    `;
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        if (feedbackEl) {
            feedbackEl.remove();
        }
    }, 3000);
}

function getCacheInfo() {
    try {
        const keys = Object.keys(localStorage);
        const cotKeys = keys.filter(key => key.startsWith('cot_analysis_'));
        const cacheInfo = cotKeys.map(key => {
            const cached = localStorage.getItem(key);
            if (cached) {
                const data = JSON.parse(cached);
                return {
                    jobId: data.jobId,
                    age: Math.round((Date.now() - data.timestamp) / 60000), // age in minutes
                    key: key
                };
            }
            return null;
        }).filter(Boolean);
        
        return cacheInfo;
    } catch (error) {
        console.error('Error getting cache info:', error);
        return [];
    }
}

function loadCachedCoTDataOnStartup() {
    try {
        const cacheInfo = getCacheInfo();
        if (cacheInfo.length > 0) {
            console.log(`Found ${cacheInfo.length} cached CoT analyses on startup`);
            
            // If there's only one cached analysis, auto-load it
            if (cacheInfo.length === 1) {
                const cachedJobId = cacheInfo[0].jobId;
                console.log(`Auto-loading cached analysis for job: ${cachedJobId}`);
                
                // Set the job select value if possible
                const jobSelect = document.getElementById('cot-job-select');
                if (jobSelect) {
                    // Find the option that matches this job ID
                    for (let option of jobSelect.options) {
                        if (option.value === cachedJobId) {
                            jobSelect.value = cachedJobId;
                            break;
                        }
                    }
                    
                    // Enable the analyze button
                    const analyzeBtn = document.getElementById('analyze-btn');
                    if (analyzeBtn && cachedJobId) {
                        analyzeBtn.disabled = false;
                    }
                    
                    // Load the cached data
                    const cachedData = getCachedCoTData(cachedJobId);
                    if (cachedData) {
                        currentCoTData = cachedData;
                        showCoTResults(cachedData, true); // true indicates cached data
                        console.log('Auto-loaded cached CoT analysis');
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error loading cached CoT data on startup:', error);
    }
}

// ===== CACHED RESULTS BROWSER FUNCTIONS =====

function showCachedResults() {
    console.log('Opening cached results browser...');
    
    const modal = document.getElementById('cached-results-modal');
    if (!modal) {
        console.error('Cached results modal not found');
        return;
    }
    
    // Show the modal
    modal.classList.remove('hidden');
    
    // Populate the cache summary and list
    populateCachedResultsList();
}

function hideCachedResults() {
    const modal = document.getElementById('cached-results-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function populateCachedResultsList() {
    const cacheInfo = getCacheInfo();
    const summaryDiv = document.getElementById('cache-summary');
    const listDiv = document.getElementById('cached-results-list');
    
    if (!summaryDiv || !listDiv) {
        console.error('Cache summary or list elements not found');
        return;
    }
    
    // Update summary
    const totalCached = cacheInfo.length;
    const totalSize = cacheInfo.reduce((total, info) => {
        const cached = localStorage.getItem(info.key);
        return total + (cached ? cached.length : 0);
    }, 0);
    
    summaryDiv.innerHTML = `
        <div class="grid grid-cols-2 gap-4 text-sm">
            <div>
                <span class="font-medium">Total Cached:</span> ${totalCached} analyses
            </div>
            <div>
                <span class="font-medium">Total Size:</span> ${(totalSize / 1024).toFixed(1)} KB
            </div>
        </div>
    `;
    
    // Update list
    if (cacheInfo.length === 0) {
        listDiv.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <div class="text-4xl mb-2">📭</div>
                <p>No cached analyses found</p>
                <p class="text-sm">Run some CoT analyses to see them cached here</p>
            </div>
        `;
        return;
    }
    
    // Sort by timestamp (newest first) - need to get full data for sorting
    const fullCacheInfo = cacheInfo.map(info => {
        const cached = localStorage.getItem(info.key);
        if (cached) {
            const data = JSON.parse(cached);
            return {
                ...info,
                timestamp: data.timestamp,
                age: Date.now() - data.timestamp
            };
        }
        return null;
    }).filter(Boolean);
    
    fullCacheInfo.sort((a, b) => b.timestamp - a.timestamp);
    
    listDiv.innerHTML = fullCacheInfo.map(info => {
        const ageMinutes = Math.floor(info.age / 60000);
        const ageHours = Math.floor(ageMinutes / 60);
        const ageDays = Math.floor(ageHours / 24);
        
        let ageText;
        if (ageDays > 0) {
            ageText = `${ageDays}d ${ageHours % 24}h ago`;
        } else if (ageHours > 0) {
            ageText = `${ageHours}h ${ageMinutes % 60}m ago`;
        } else {
            ageText = `${ageMinutes}m ago`;
        }
        
        return `
            <div class="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                <div class="flex items-center justify-between">
                    <div class="flex-1">
                        <div class="flex items-center space-x-3">
                            <div class="w-3 h-3 bg-green-400 rounded-full"></div>
                            <div>
                                <h4 class="font-medium text-gray-900">Job ${info.jobId}</h4>
                                <p class="text-sm text-gray-500">Cached ${ageText}</p>
                            </div>
                        </div>
                    </div>
                    <div class="flex space-x-2">
                        <button onclick="loadCachedAnalysis('${info.jobId}')" 
                            class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">
                            📖 View
                        </button>
                        <button onclick="deleteCachedAnalysis('${info.jobId}')" 
                            class="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function loadCachedAnalysis(jobId) {
    console.log(`Loading cached analysis for job: ${jobId}`);
    
    const cachedData = getCachedCoTData(jobId);
    if (!cachedData) {
        showCoTCacheFeedback('Cached data not found or expired');
        return;
    }
    
    // Close the modal
    hideCachedResults();
    
    // Set the job in the dropdown
    const jobSelect = document.getElementById('cot-job-select');
    if (jobSelect) {
        jobSelect.value = jobId;
    }
    
    // Load and display the cached data
    currentCoTData = cachedData;
    showCoTResults(cachedData, true); // true indicates cached data
    
    showCoTCacheFeedback(`Loaded cached analysis for job ${jobId}`);
}

function deleteCachedAnalysis(jobId) {
    if (confirm(`Are you sure you want to delete the cached analysis for job ${jobId}?`)) {
        clearCoTCache(jobId);
        populateCachedResultsList(); // Refresh the list
    }
}

function clearAllCachedResults() {
    const cacheInfo = getCacheInfo();
    if (cacheInfo.length === 0) {
        showCoTCacheFeedback('No cached analyses to clear');
        return;
    }
    
    if (confirm(`Are you sure you want to delete all ${cacheInfo.length} cached analyses?`)) {
        clearCoTCache(); // Clear all
        populateCachedResultsList(); // Refresh the list
        hideCachedResults(); // Close modal
    }
}

// Close modal when clicking outside
document.addEventListener('click', function(event) {
    const modal = document.getElementById('cached-results-modal');
    if (modal && !modal.classList.contains('hidden')) {
        if (event.target === modal) {
            hideCachedResults();
        }
    }
});

// ============================================================================
// HEATMAP VISUALIZATION
// ============================================================================

// Load jobs that have probability tracking enabled
async function loadHeatmapJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        
        const jobSelect = document.getElementById('heatmap-job-select');
        jobSelect.innerHTML = '<option value="">Select a completed job...</option>';
        
        // Filter for completed jobs with prob_file
        const jobsWithProb = data.jobs.filter(job => 
            job.status === 'DONE' && 
            job.prob_file && 
            job.prob_file !== null
        );
        
        jobsWithProb.forEach(job => {
            const option = document.createElement('option');
            option.value = job.job_id;
            const model = job.request?.model || 'Unknown';
            const dataset = job.request?.dataset || 'Unknown';
            option.textContent = `${job.job_id.substring(0, 8)} - ${model} - ${dataset}`;
            jobSelect.appendChild(option);
        });
        
        if (jobsWithProb.length === 0) {
            showHeatmapError('No completed jobs with probability tracking found. Submit a job with "Enable Probability Tracking" checked.');
        }
    } catch (error) {
        console.error('Error loading heatmap jobs:', error);
        showHeatmapError('Failed to load jobs: ' + error.message);
    }
}

// Load questions for selected job
async function loadHeatmapQuestions() {
    const jobId = document.getElementById('heatmap-job-select').value;
    const questionSelect = document.getElementById('heatmap-question-select');
    
    if (!jobId) {
        questionSelect.innerHTML = '<option value="">Select a question...</option>';
        questionSelect.disabled = true;
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}/questions`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        const questions = await response.json();
        questionSelect.innerHTML = '<option value="">Select a question...</option>';
        
        questions.forEach(q => {
            const option = document.createElement('option');
            option.value = q.idx;
            option.textContent = `Question ${q.idx}: ${q.preview}`;
            if (!q.has_prob_data) {
                option.textContent += ' (No prob data)';
                option.disabled = true;
            }
            questionSelect.appendChild(option);
        });
        
        questionSelect.disabled = false;
        hideHeatmapError();
    } catch (error) {
        console.error('Error loading questions:', error);
        showHeatmapError('Failed to load questions: ' + error.message);
        questionSelect.disabled = true;
    }
}

// Load and display heatmap data
async function loadHeatmapData() {
    const jobId = document.getElementById('heatmap-job-select').value;
    const questionIdx = document.getElementById('heatmap-question-select').value;
    
    if (!jobId || questionIdx === '') {
        return;
    }
    
    // Show loading state
    document.getElementById('heatmap-loading').classList.remove('hidden');
    document.getElementById('heatmap-display').classList.add('hidden');
    document.getElementById('heatmap-question-display').classList.add('hidden');
    // Hide token info box when loading new data
    const infoBox = document.getElementById('plot-c-token-info');
    if (infoBox) {
        infoBox.classList.add('hidden');
    }
    hideHeatmapError();
    
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}/heatmap-data/${questionIdx}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        const data = await response.json();
        
        // Display question
        document.getElementById('heatmap-question-text').textContent = data.question_text;
        
        // Display answer status
        const statusElement = document.getElementById('heatmap-answer-status');
        const predictedElement = document.getElementById('heatmap-predicted-answer');
        const groundTruthElement = document.getElementById('heatmap-ground-truth');
        
        // Update status badge
        if (data.is_correct) {
            statusElement.textContent = 'CORRECT';
            statusElement.className = 'ml-2 px-2 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';
        } else {
            statusElement.textContent = 'INCORRECT';
            statusElement.className = 'ml-2 px-2 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800';
        }
        
        // Update predicted and ground truth answers
        predictedElement.textContent = data.predicted_answer || 'N/A';
        groundTruthElement.textContent = data.ground_truth || 'N/A';
        
        document.getElementById('heatmap-question-display').classList.remove('hidden');
        
        // Update heatmap panel titles with correctness info
        const correctTitle = document.querySelector('#heatmap-correct .text-center h3');
        const chosenTitle = document.querySelector('#heatmap-chosen .text-center h3');
        
        if (correctTitle) {
            const statusText = data.is_correct ? '✓' : '✗';
            correctTitle.innerHTML = `Ground Truth Token Probabilities <span class="ml-2 text-sm ${data.is_correct ? 'text-green-600' : 'text-red-600'}">${statusText}</span>`;
            correctTitle.title = "Shows the probability of the ground truth token at each generation step. Low probabilities indicate the model was not confident in the correct answer. Note: Due to tokenization differences, some probabilities may appear unusually high.";
        }
        
        if (chosenTitle) {
            const statusText = data.is_correct ? '✓' : '✗';
            chosenTitle.innerHTML = `Chosen Token Probabilities <span class="ml-2 text-sm ${data.is_correct ? 'text-green-600' : 'text-red-600'}">${statusText}</span>`;
            chosenTitle.title = "Shows the probability of the token the model actually chose at each generation step. High probabilities indicate the model was confident in its choice.";
        }
        
        // Render both heatmaps
        renderHeatmap(data.output_tokens, data.chosen_probs, 'heatmap-chosen');
        renderHeatmap(data.output_tokens, data.correct_probs, 'heatmap-correct');
        
        // Render probability plots (pass original probabilities if available)
        renderProbabilityPlots(
            data.output_tokens, 
            data.chosen_probs, 
            data.correct_probs,
            data.chosen_probs_original || data.chosen_probs,
            data.correct_probs_original || data.correct_probs
        );
        
        // Show heatmap display
        document.getElementById('heatmap-loading').classList.add('hidden');
        document.getElementById('heatmap-display').classList.remove('hidden');
        
    } catch (error) {
        console.error('Error loading heatmap data:', error);
        document.getElementById('heatmap-loading').classList.add('hidden');
        showHeatmapError('Failed to load heatmap data: ' + error.message);
    }
}

// Render heatmap with colored tokens
function renderHeatmap(tokens, probs, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (!tokens || tokens.length === 0) {
        container.innerHTML = '<p class="text-gray-500">No tokens to display</p>';
        return;
    }
    
    tokens.forEach((token, idx) => {
        const prob = probs[idx] || 0;
        const color = getHeatmapColor(prob);
        
        const span = document.createElement('span');
        span.textContent = token;
        span.style.backgroundColor = color;
        span.style.padding = '2px 4px';
        span.style.margin = '2px';
        span.style.borderRadius = '3px';
        span.style.display = 'inline-block';
        span.style.border = '1px solid #e5e7eb';
        span.title = `Probability: ${(prob * 100).toFixed(2)}%`;
        
        container.appendChild(span);
    });
}

// Convert probability (0-1) to white-to-red color
function getHeatmapColor(prob) {
    // Clamp probability between 0 and 1
    prob = Math.max(0, Math.min(1, prob));
    // Interpolate from white (low) to red (high)
    const r = 255;
    const g = Math.round(255 * (1 - prob));
    const b = Math.round(255 * (1 - prob));
    return `rgb(${r}, ${g}, ${b})`;
}

function renderProbabilityPlots(tokens, chosenProbs, correctProbs, chosenProbsOriginal, correctProbsOriginal) {
    // Show plots section
    document.getElementById('heatmap-plots-section').classList.remove('hidden');
    
    // Plot A: Line chart showing probability over token positions
    renderLineChart(tokens, chosenProbs, correctProbs);
    
    // Plot B: Bar chart comparing average probabilities
    renderBarChart(chosenProbs, correctProbs);
    
    // Plot C: Confidence trend (moving average)
    renderConfidenceTrend(tokens, chosenProbs, correctProbs, chosenProbsOriginal, correctProbsOriginal);
}

function renderLineChart(tokens, chosenProbs, correctProbs) {
    const ctx = document.getElementById('plot-a-canvas').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.plotAChart) {
        window.plotAChart.destroy();
    }
    
    window.plotAChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: tokens.map((_, idx) => idx),
            datasets: [
                {
                    label: 'Chosen Token Probability',
                    data: chosenProbs,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'Ground Truth Token Probability',
                    data: correctProbs,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Probability Distribution Across Token Positions' }
            },
            scales: {
                y: { beginAtZero: true, max: 1, title: { display: true, text: 'Probability' } },
                x: { title: { display: true, text: 'Token Position' } }
            }
        }
    });
}

function renderBarChart(chosenProbs, correctProbs) {
    const ctx = document.getElementById('plot-b-canvas').getContext('2d');
    
    if (window.plotBChart) {
        window.plotBChart.destroy();
    }
    
    const avgChosen = chosenProbs.reduce((a, b) => a + b, 0) / chosenProbs.length;
    const avgCorrect = correctProbs.reduce((a, b) => a + b, 0) / correctProbs.length;
    const minChosen = Math.min(...chosenProbs);
    const maxChosen = Math.max(...chosenProbs);
    const minCorrect = Math.min(...correctProbs);
    const maxCorrect = Math.max(...correctProbs);
    
    window.plotBChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Average', 'Minimum', 'Maximum'],
            datasets: [
                {
                    label: 'Chosen Token',
                    data: [avgChosen, minChosen, maxChosen],
                    backgroundColor: 'rgba(59, 130, 246, 0.7)',
                    borderColor: 'rgb(59, 130, 246)',
                    borderWidth: 1
                },
                {
                    label: 'Ground Truth Token',
                    data: [avgCorrect, minCorrect, maxCorrect],
                    backgroundColor: 'rgba(239, 68, 68, 0.7)',
                    borderColor: 'rgb(239, 68, 68)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Probability Statistics Comparison' }
            },
            scales: {
                y: { beginAtZero: true, max: 1, title: { display: true, text: 'Probability' } }
            }
        }
    });
}

function renderConfidenceTrend(tokens, chosenProbs, correctProbs, chosenProbsOriginal, correctProbsOriginal) {
    const ctx = document.getElementById('plot-c-canvas').getContext('2d');
    
    if (window.plotCChart) {
        window.plotCChart.destroy();
    }
    
    // Calculate moving average with window size 10
    const windowSize = 10;
    const movingAvgChosen = calculateMovingAverage(chosenProbs, windowSize);
    const movingAvgCorrect = calculateMovingAverage(correctProbs, windowSize);
    
    // Store data for click handler access
    const chartData = {
        tokens: tokens,
        chosenProbsOriginal: chosenProbsOriginal,
        correctProbsOriginal: correctProbsOriginal
    };
    
    window.plotCChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: movingAvgChosen.map((_, idx) => idx),
            datasets: [
                {
                    label: 'Chosen Token Confidence (Moving Avg)',
                    data: movingAvgChosen,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Ground Truth Confidence (Moving Avg)',
                    data: movingAvgCorrect,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Confidence Trend (10-token Moving Average)' }
            },
            scales: {
                y: { beginAtZero: true, max: 1, title: { display: true, text: 'Confidence' } },
                x: { title: { display: true, text: 'Position' } }
            },
            onClick: (event, activeElements) => {
                let clickedIndex = null;
                
                // If clicking directly on a data point, use that index
                if (activeElements.length > 0) {
                    clickedIndex = activeElements[0].index;
                } else {
                    // If clicking on chart area, find the nearest data point
                    const canvasPosition = Chart.helpers.getRelativePosition(event, window.plotCChart);
                    const dataX = window.plotCChart.scales.x.getValueForPixel(canvasPosition.x);
                    // Round to nearest integer index
                    clickedIndex = Math.round(dataX);
                    // Clamp to valid range
                    clickedIndex = Math.max(0, Math.min(chartData.tokens.length - 1, clickedIndex));
                }
                
                if (clickedIndex !== null && clickedIndex >= 0 && clickedIndex < chartData.tokens.length) {
                    displayTokenWindowInfo(clickedIndex, chartData.tokens, chartData.chosenProbsOriginal, chartData.correctProbsOriginal);
                }
            }
        }
    });
}

function calculateMovingAverage(data, windowSize) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - Math.floor(windowSize / 2));
        const end = Math.min(data.length, i + Math.ceil(windowSize / 2));
        const window = data.slice(start, end);
        const avg = window.reduce((a, b) => a + b, 0) / window.length;
        result.push(avg);
    }
    return result;
}

function displayTokenWindowInfo(clickedIndex, tokens, chosenProbsOriginal, correctProbsOriginal) {
    // Calculate window bounds: center position ± 5 tokens (window size 10)
    const windowSize = 10;
    const halfWindow = Math.floor(windowSize / 2);
    const start = Math.max(0, clickedIndex - halfWindow);
    const end = Math.min(tokens.length, clickedIndex + halfWindow + 1);
    
    // Get the info box elements
    const infoBox = document.getElementById('plot-c-token-info');
    const infoBody = document.getElementById('plot-c-token-info-body');
    
    if (!infoBox || !infoBody) {
        console.error('Info box elements not found');
        return;
    }
    
    // Clear previous content
    infoBody.innerHTML = '';
    
    // Update header to show clicked position
    const header = infoBox.querySelector('h5');
    if (header) {
        header.textContent = `Token Details at Position ${clickedIndex} (Window: positions ${start} to ${end - 1})`;
    }
    
    // Create rows for each token in the window
    for (let i = start; i < end; i++) {
        const row = document.createElement('tr');
        
        // Highlight the center token (clicked position)
        if (i === clickedIndex) {
            row.className = 'bg-blue-50 font-medium';
        } else {
            row.className = 'border-b border-gray-200';
        }
        
        // Position column
        const posCell = document.createElement('td');
        posCell.className = 'px-3 py-2 text-gray-900';
        posCell.textContent = i;
        row.appendChild(posCell);
        
        // Token text column (escape HTML and handle special characters)
        const tokenCell = document.createElement('td');
        tokenCell.className = 'px-3 py-2 text-gray-900 font-mono';
        const tokenText = tokens[i] || '';
        // Escape HTML and handle whitespace characters
        const escapedToken = tokenText
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;')
            .replace(/\n/g, '↵')
            .replace(/\t/g, '⇥')
            .replace(/ /g, '␣');
        tokenCell.innerHTML = escapedToken || '<span class="text-gray-400">(empty)</span>';
        row.appendChild(tokenCell);
        
        // Chosen probability column
        const chosenProbCell = document.createElement('td');
        chosenProbCell.className = 'px-3 py-2 text-gray-700';
        const chosenProb = chosenProbsOriginal && chosenProbsOriginal[i] !== undefined 
            ? chosenProbsOriginal[i] 
            : null;
        if (chosenProb !== null) {
            chosenProbCell.textContent = chosenProb.toFixed(6);
        } else {
            chosenProbCell.textContent = 'N/A';
            chosenProbCell.className += ' text-gray-400';
        }
        row.appendChild(chosenProbCell);
        
        // Correct probability column
        const correctProbCell = document.createElement('td');
        correctProbCell.className = 'px-3 py-2 text-gray-700';
        const correctProb = correctProbsOriginal && correctProbsOriginal[i] !== undefined 
            ? correctProbsOriginal[i] 
            : null;
        if (correctProb !== null) {
            correctProbCell.textContent = correctProb.toFixed(6);
        } else {
            correctProbCell.textContent = 'N/A';
            correctProbCell.className += ' text-gray-400';
        }
        row.appendChild(correctProbCell);
        
        infoBody.appendChild(row);
    }
    
    // Show the info box
    infoBox.classList.remove('hidden');
}

// Show/hide error messages
function showHeatmapError(message) {
    const errorDiv = document.getElementById('heatmap-error');
    const errorMessage = document.getElementById('heatmap-error-message');
    errorMessage.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideHeatmapError() {
    document.getElementById('heatmap-error').classList.add('hidden');
}

function togglePlot(plotId) {
    const plot = document.getElementById(plotId);
    const button = document.getElementById(`toggle-${plotId}`);
    
    if (plot.classList.contains('hidden')) {
        plot.classList.remove('hidden');
        button.classList.remove('bg-blue-600', 'hover:bg-blue-700');
        button.classList.add('bg-green-600', 'hover:bg-green-700');
    } else {
        plot.classList.add('hidden');
        button.classList.remove('bg-green-600', 'hover:bg-green-700');
        button.classList.add('bg-blue-600', 'hover:bg-blue-700');
        
        // Hide token info box when plot-c is hidden
        if (plotId === 'plot-c') {
            const infoBox = document.getElementById('plot-c-token-info');
            if (infoBox) {
                infoBox.classList.add('hidden');
            }
        }
    }
}

// Initialize heatmap tab on load
document.addEventListener('DOMContentLoaded', function() {
    // Jobs are now loaded automatically when tabs are switched in showTab function
});