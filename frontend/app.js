// Configuration
const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// Available options
const AVAILABLE_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B',
    'Qwen/Qwen2.5-Math-7B',
    'Qwen/Qwen2.5-Math-14B',
    'Qwen/Qwen2.5-Math-72B'
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

// Tab management
function showTab(tabName) {
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
    document.getElementById(tabName).classList.add('active');

    // Highlight selected tab button
    event.target.classList.remove('text-gray-500', 'hover:text-gray-700');
    event.target.classList.add('bg-blue-600', 'text-white');

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

        const response = await fetch(`${API_BASE}/config/paths`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();
        if (response.ok) {
            alert('Path configuration saved successfully!');
            console.log('Path configuration saved:', data);
        } else {
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
        n_sampling: '1',
        seed: '42',
        eval_method: EVAL_METHODS[0],
        k: '1',
        max_tokens: '2048',
        prompt: 'Solve this math problem step by step: {question}'
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

function calculateJobCount() {
    let totalJobs = 0;
    modelConfigs.forEach(config => {
        const models = config.customModel ? [config.customModel] : [config.model];
        const datasets = config.dataset.split('\n').filter(d => d.trim());
        const temperatures = config.temperature.split('\n').filter(t => t.trim());
        const top_ps = config.top_p.split('\n').filter(t => t.trim());
        const seeds = config.seed.split('\n').filter(s => s.trim());
        const n_samplings = config.n_sampling.split('\n').filter(n => n.trim());
        const ks = config.k.split('\n').filter(k => k.trim());
        const max_tokens = config.max_tokens.split('\n').filter(m => m.trim());

        const combinations = models.length * datasets.length * temperatures.length *
            top_ps.length * seeds.length * n_samplings.length * ks.length * max_tokens.length;
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
                    <select onchange="updateModelConfig(${config.id}, 'model', this.value); updateModelConfig(${config.id}, 'customModel', '')" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${AVAILABLE_MODELS.map(model => `<option value="${model}" ${config.model === model ? 'selected' : ''}>${model}</option>`).join('')}
                    </select>
                    <input type="text" onchange="updateModelConfig(${config.id}, 'customModel', this.value); updateModelConfig(${config.id}, 'model', '')" 
                           placeholder="Or enter custom Hugging Face model URL..." 
                           class="mt-2 block w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                           value="${config.customModel}">
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
                    <label class="block text-sm font-medium text-gray-700">N Sampling (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'n_sampling', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="1&#10;5&#10;10">${config.n_sampling}</textarea>
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
                    <label class="block text-sm font-medium text-gray-700">K (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'k', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="1&#10;5&#10;10">${config.k}</textarea>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Max Tokens (one per line)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'max_tokens', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="2048\n4096">${config.max_tokens}</textarea>
                </div>
                
                <div class="md:col-span-2 lg:col-span-3">
                    <label class="block text-sm font-medium text-gray-700">Custom Prompt Template (Required)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'prompt', this.value)" 
                              class="mt-1 block w-full border border-blue-300 rounded-md px-3 py-2 bg-blue-50" 
                              rows="6" placeholder="Enter your custom prompt template here...">${config.prompt}</textarea>
                    <p class="text-xs text-blue-600 mt-1 font-medium">
                        ⚡ Use {question} to insert the math problem. Example: "Solve this math problem step by step: {question}"
                    </p>
                </div>
            </div>
        `;
        container.appendChild(configDiv);
    });
    updateJobCount();
}

// API functions
async function submitEvaluation() {
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
            const models = config.customModel ? [config.customModel] : [config.model];
            const datasets = config.dataset.split('\n').filter(d => d.trim());
            const temperatures = config.temperature.split('\n').filter(t => t.trim());
            const top_ps = config.top_p.split('\n').filter(t => t.trim());
            const top_ks = config.top_k.split('\n').filter(k => k.trim());
            const seeds = config.seed.split('\n').filter(s => s.trim());
            const n_samplings = config.n_sampling.split('\n').filter(n => n.trim());
            const ks = config.k.split('\n').filter(k => k.trim());
            const max_tokens = config.max_tokens.split('\n').filter(m => m.trim());

            // Generate all combinations
            models.forEach(model => {
                datasets.forEach(dataset => {
                    temperatures.forEach(temp => {
                        top_ps.forEach(top_p => {
                            top_ks.forEach(top_k => {
                                seeds.forEach(seed => {
                                    n_samplings.forEach(n_sampling => {
                                        ks.forEach(k => {
                                            max_tokens.forEach(max_token => {
                                                // Validate that prompt is provided
                                                if (!config.prompt || config.prompt.trim() === '') {
                                                    throw new Error(`Prompt is required for model configuration ${config.id}`);
                                                }
                                                
                                                const requestData = {
                                                    model: model,
                                                    dataset: dataset,
                                                    prompt: config.prompt.trim(),
                                                    backend: config.backend,
                                                    temperature: parseFloat(temp),
                                                    top_p: parseFloat(top_p),
                                                    top_k: parseInt(top_k),
                                                    n_sampling: parseInt(n_sampling),
                                                    seed: parseInt(seed),
                                                    eval_method: config.eval_method,
                                                    k: parseInt(k),
                                                    max_tokens: parseInt(max_token)
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
        // Compute metrics file path
        let metricsFile = '';
        if (job.result_file) {
            // Handle both old format (with prompt_type in filename) and new format (custom)
            if (job.result_file.includes('tool-integrated')) {
                // Old format: test_tool-integrated_-1_seed42_t1.0_s0_e-1.jsonl -> test_tool-integrated_-1_seed42_t1.0_s0_e-1_tool-integrated_metrics.json
                metricsFile = job.result_file.replace(/(\.jsonl|\.json)$/g, `_tool-integrated_metrics.json`);
            } else {
                // New format: test_custom_-1_seed42_t0.0_s0_e-1.jsonl -> test_custom_-1_seed42_t0.0_s0_e-1_metrics.json
                metricsFile = job.result_file.replace(/(\.jsonl|\.json)$/g, `_metrics.json`);
            }
        }
        // Buttons for viewing results
        let resultButtonHtml = '';
        if (job.status === 'DONE' && job.result_file) {
            resultButtonHtml = `
                <div class="flex space-x-2 mt-4">
                    <button onclick="showResultModal('${metricsFile}', 'Metrics')" class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">View Metrics Inline</button>
                    <button onclick="showResultModal('${job.result_file}', 'Model Outputs')" class="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700">View Model Outputs Inline</button>
                </div>
            `;
        }
        // Delete button always at top right
        const deleteButtonHtml = `
            <button onclick="deleteJob('${job.job_id}')" title="Delete job" class="absolute top-2 right-2 px-2 py-1 text-red-600 hover:text-red-800 z-10">
                <svg xmlns='http://www.w3.org/2000/svg' class='h-5 w-5' fill='none' viewBox='0 0 24 24' stroke='currentColor'><path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M6 18L18 6M6 6l12 12'/></svg>
            </button>
        `;
        // Monitor button only for non-completed jobs
        let monitorButtonHtml = '';
        if (job.status !== 'DONE' && job.status !== 'ERROR') {
            monitorButtonHtml = `<button onclick="monitorJob('${job.job_id}')" class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">Monitor</button>`;
        }
        jobDiv.innerHTML = `
            ${deleteButtonHtml}
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
            if (data.out) {
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

async function showResultModal(resultFilePath, title = 'Results') {
    try {
        const url = `http://localhost:8000/file?path=${encodeURIComponent(resultFilePath)}`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const content = await response.text();
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-lg p-6 max-w-4xl max-h-96 overflow-auto">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold">${title}: ${resultFilePath.split('/').pop()}</h3>
                    <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700 text-xl">&times;</button>
                </div>
                <pre class="text-sm bg-gray-100 p-4 rounded overflow-auto max-h-64">${escapeHtml(content)}</pre>
            </div>
        `;
        document.body.appendChild(modal);
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    } catch (error) {
        console.error('Error loading result file:', error);
        alert('Error loading result file: ' + error.message);
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
}); 