// Configuration
const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// Available options
const AVAILABLE_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B',
    'Qwen/Qwen2.5-Math-3B',
    'Qwen/Qwen2.5-Math-7B',
    'Qwen/Qwen2.5-Math-14B'
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

const BACKEND_OPTIONS = [
    'local',
    'bash',
    'slurm'
];

// Global state
let modelConfigs = [];
let currentWebSocket = null;

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
}

// Model configuration management
function addModelConfig() {
    const configId = Date.now();
    const config = {
        id: configId,
        model: AVAILABLE_MODELS[0],
        dataset: AVAILABLE_DATASETS[0],
        backend: BACKEND_OPTIONS[0],
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        n_sampling: 1,
        seed: 42,
        eval_method: EVAL_METHODS[0],
        k: 1,
        prompt: ''
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
                    <label class="block text-sm font-medium text-gray-700">Model</label>
                    <select onchange="updateModelConfig(${config.id}, 'model', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${AVAILABLE_MODELS.map(model => `<option value="${model}" ${config.model === model ? 'selected' : ''}>${model}</option>`).join('')}
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Dataset</label>
                    <select onchange="updateModelConfig(${config.id}, 'dataset', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${AVAILABLE_DATASETS.map(dataset => `<option value="${dataset}" ${config.dataset === dataset ? 'selected' : ''}>${dataset}</option>`).join('')}
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Backend</label>
                    <select onchange="updateModelConfig(${config.id}, 'backend', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${BACKEND_OPTIONS.map(backend => `<option value="${backend}" ${config.backend === backend ? 'selected' : ''}>${backend}</option>`).join('')}
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Temperature</label>
                    <input type="range" min="0" max="5" step="0.1" value="${config.temperature}" 
                           onchange="updateModelConfig(${config.id}, 'temperature', parseFloat(this.value))" 
                           class="mt-1 block w-full">
                    <span class="text-sm text-gray-500">${config.temperature}</span>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Top P</label>
                    <input type="range" min="0" max="1" step="0.1" value="${config.top_p}" 
                           onchange="updateModelConfig(${config.id}, 'top_p', parseFloat(this.value))" 
                           class="mt-1 block w-full">
                    <span class="text-sm text-gray-500">${config.top_p}</span>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Top K</label>
                    <input type="number" min="0" value="${config.top_k}" 
                           onchange="updateModelConfig(${config.id}, 'top_k', parseInt(this.value))" 
                           class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">N Sampling</label>
                    <input type="number" min="1" max="32" value="${config.n_sampling}" 
                           onchange="updateModelConfig(${config.id}, 'n_sampling', parseInt(this.value))" 
                           class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Seed</label>
                    <input type="number" min="0" value="${config.seed}" 
                           onchange="updateModelConfig(${config.id}, 'seed', parseInt(this.value))" 
                           class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">Eval Method</label>
                    <select onchange="updateModelConfig(${config.id}, 'eval_method', this.value)" class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                        ${EVAL_METHODS.map(method => `<option value="${method}" ${config.eval_method === method ? 'selected' : ''}>${method}</option>`).join('')}
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-700">K</label>
                    <input type="number" min="1" max="32" value="${config.k}" 
                           onchange="updateModelConfig(${config.id}, 'k', parseInt(this.value))" 
                           class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2">
                </div>
                
                <div class="md:col-span-2 lg:col-span-3">
                    <label class="block text-sm font-medium text-gray-700">Prompt (Optional)</label>
                    <textarea onchange="updateModelConfig(${config.id}, 'prompt', this.value)" 
                              class="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2" 
                              rows="3" placeholder="Custom prompt template...">${config.prompt}</textarea>
                </div>
            </div>
        `;
        container.appendChild(configDiv);
    });
}

// API functions
async function submitEvaluation() {
    if (modelConfigs.length === 0) {
        alert('Please add at least one model configuration');
        return;
    }
    
    try {
        const promises = modelConfigs.map(config => {
            const requestData = {
                model: config.model,
                dataset: config.dataset,
                backend: config.backend,
                temperature: config.temperature,
                top_p: config.top_p,
                top_k: config.top_k,
                n_sampling: config.n_sampling,
                seed: config.seed,
                eval_method: config.eval_method,
                k: config.k
            };
            
            if (config.prompt) {
                requestData.prompt = config.prompt;
            }
            
            return fetch(`${API_BASE}/jobs`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            }).then(response => response.json());
        });
        
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

async function refreshJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        const jobs = data.jobs || [];
        const jobsList = document.getElementById('jobs-list');
        jobsList.innerHTML = '';
        if (jobs.length === 0) {
            jobsList.innerHTML = '<p class="text-gray-500">No jobs found</p>';
            return;
        }
        jobs.forEach(job => {
            const jobDiv = document.createElement('div');
            jobDiv.className = 'border border-gray-200 rounded-lg p-4 mb-4';
            jobDiv.innerHTML = `
                <div class="flex justify-between items-start">
                    <div>
                        <h3 class="font-medium">Job ID: ${job.job_id}</h3>
                        <p class="text-sm text-gray-600">Model: ${job.request?.model || 'N/A'}</p>
                        <p class="text-sm text-gray-600">Dataset: ${job.request?.dataset || 'N/A'}</p>
                        <p class="text-sm text-gray-600">Status: <span class="font-medium ${getStatusColor(job.status)}">${job.status}</span></p>
                    </div>
                    <div class="flex space-x-2">
                        <button onclick="monitorJob('${job.job_id}')" class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700">
                            Monitor
                        </button>
                    </div>
                </div>
            `;
            jobsList.appendChild(jobDiv);
        });
    } catch (error) {
        console.error('Error fetching jobs:', error);
        document.getElementById('jobs-list').innerHTML = '<p class="text-red-500">Error loading jobs</p>';
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

function monitorJob(jobId) {
    document.getElementById('monitor-job-id').value = jobId;
    showTab('monitor');
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
    
    currentWebSocket.onopen = function() {
        logOutput.innerHTML += 'Connected to job stream\n';
    };
    
    currentWebSocket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.log) {
                logOutput.innerHTML += data.log;
                logOutput.scrollTop = logOutput.scrollHeight;
            } else if (data.gpu) {
                logOutput.innerHTML += `[GPU] Memory: ${Math.round(data.gpu.mem / 1024 / 1024)}MB, Utilization: ${data.gpu.util}%\n`;
                logOutput.scrollTop = logOutput.scrollHeight;
            }
        } catch (error) {
            logOutput.innerHTML += event.data + '\n';
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    };
    
    currentWebSocket.onerror = function(error) {
        logOutput.innerHTML += 'WebSocket error: ' + error + '\n';
    };
    
    currentWebSocket.onclose = function() {
        logOutput.innerHTML += 'WebSocket connection closed\n';
    };
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Add initial model config
    addModelConfig();
    
    // Load jobs on page load
    refreshJobs();
}); 