# Reasoning Model Evaluator

A comprehensive web-based evaluation platform for reasoning models, built on top of the Qwen2.5-Math evaluation framework. This tool provides a lightweight web UI to configure hyperparameters, run evaluations on local machines or SLURM clusters, and monitor job progress in real-time.

## Features

- **🎯 Multi-Parameter Configuration**: Configure temperature, top-p, top-k, n_sampling, seeds, max_tokens with multiple values per line for batch evaluation
- **🤖 Model Flexibility**: Use predefined models or custom Hugging Face model URLs
- **📊 Dataset Support**: Use built-in datasets (gsm8k, math, etc.) or custom Hugging Face dataset URLs
- **📝 Prompt Preview**: Preview exact prompts with few-shot examples before running evaluations
- **⚡ Real-time Monitoring**: Live log streaming with WebSocket support for both local and SLURM jobs
- **🔧 Job Management**: View, cancel, delete, and monitor jobs with persistent storage
- **📈 Result Analysis**: Direct access to evaluation results, probability plots, and CoT analysis
- **☁️ SLURM Integration**: Full SLURM cluster support with automatic job status updates
- **💾 Persistent Storage**: Jobs are saved to disk and restored on backend restart

## Architecture

- **Backend**: FastAPI application that wraps the existing `math_eval.py` evaluation script
- **Frontend**: HTML/JavaScript UI for parameter configuration and real-time monitoring
- **Runner**: Async job management and progress streaming with UUID-based tracking
- **Path Manager**: Centralized path configuration with automatic detection and validation

## Directory Structure

```
reasoning-models-eval/
├── evaluation/              # Original Qwen2.5-Math evaluation code
│   ├── math_eval.py        # Main evaluation script
│   ├── togetherapi.py      # Together AI API integration
│   ├── trunc_plot.py       # Truncation analysis plotting
│   ├── requirements.txt    # Evaluation dependencies
│   └── ...
├── backend/                # FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI entrypoint with job management APIs
│   │   ├── schemas.py      # Pydantic models for requests/responses
│   │   ├── runner.py       # Job execution wrapper for math_eval.py
│   │   ├── path_manager.py # Path configuration management
│   │   └── enums.py        # Backend type definitions
│   ├── tests/              # Test files
│   ├── requirements.txt    # Backend dependencies
│   └── path_config.json    # Path configuration (user-specific)
├── frontend/               # Web UI
│   ├── index.html          # Main UI page with tabs
│   ├── app.js              # JavaScript functionality
│   └── config.js           # Auto-generated API configuration
├── start.sh                # Startup script
├── README.md               # This file
└── ...
```

## Prerequisites

- **Python**: Python 3.10 or higher (3.10-3.13 recommended)
- **Conda**: Miniconda or Anaconda for environment management
- **CUDA**: CUDA-compatible GPU (for local GPU execution)
- **SLURM**: SLURM cluster access (optional, for cluster execution)
- **Git**: For cloning the repository

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HumainLab/benchmark_reproducibility.git reasoning-models-eval
cd reasoning-models-eval
```

Or if cloning from the original Qwen repository:

```bash
git clone https://github.com/QwenLM/Qwen2.5-Math.git reasoning-models-eval
cd reasoning-models-eval
```

### 2. Set Up Conda Environment

#### Option A: Create Environment in `/work` (Recommended for Cluster Systems)

If you're on a cluster system with limited home directory quota, create the environment in `/work`:

```bash
# Create conda environment in /work for disk space
conda create -y -p /work/$USER/ls6/miniconda3/envs/reasoning-models-eval python=3.10

# Activate the environment
conda activate /work/$USER/ls6/miniconda3/envs/reasoning-models-eval
```

#### Option B: Create Environment in Home Directory

For local systems or if you have sufficient home directory space:

```bash
# Create conda environment
conda create -y -n reasoning-models-eval python=3.10

# Activate the environment
conda activate reasoning-models-eval
```

**Note**: Replace `reasoning-models-eval` with your preferred environment name.

### 3. Install Backend Dependencies

Install the required packages for the FastAPI backend:

```bash
# Navigate to the project root
cd reasoning-models-eval

# Install backend dependencies
pip install -r backend/requirements.txt
```

The backend requirements include:
- `fastapi>=0.104.0` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.0.0` - Data validation
- `websockets>=12.0` - WebSocket support
- `pandas>=2.0.0` - Data manipulation
- `pynvml>=11.5.0` - GPU monitoring
- `matplotlib>=3.5.0` - Plotting
- `scikit-learn>=1.0.0` - Machine learning utilities
- `openpyxl>=3.1.0` - Excel file support
- `numpy>=1.21.0` - Numerical computing

### 4. Install Evaluation Dependencies

Install the required packages for the evaluation engine:

```bash
# Install evaluation dependencies
pip install -r evaluation/requirements.txt
```

The evaluation requirements include:
- `torch` - PyTorch for model inference
- `transformers` - Hugging Face transformers
- `vllm` - High-performance LLM inference (optional, for faster inference)
- `tqdm` - Progress bars
- `datasets` - Hugging Face datasets
- `sympy==1.12` - Symbolic mathematics
- `antlr4-python3-runtime==4.7.2` - Required by latex2sympy2
- `latex2sympy2` - LaTeX to SymPy conversion
- `word2number` - Word to number conversion
- `Pebble` - Process pool executor
- `timeout-decorator` - Timeout decorators
- `multiprocess` - Multiprocessing support
- `matplotlib` - Plotting
- `scikit-learn` - Machine learning utilities

#### Optional: Install Flash Attention (for Speed Optimization)

Flash Attention can significantly speed up inference but requires specific CUDA versions:

```bash
# Install flash-attn (optional, for speed optimization)
pip install flash-attn --no-build-isolation
```

**Note**: If `flash-attn` installation fails, you can skip it. The evaluation will work without it, but inference may be slower.

#### Optional: Install Together AI SDK (for Together API)

If you want to use Together AI's API for inference:

```bash
pip install together
```

### 5. Configure Path Settings

The application uses a path configuration file to manage directories. On first run, the backend will create a default configuration. You can customize it through the web UI:

1. Start the backend (see "Running the Application" below)
2. Navigate to the "Configure" tab in the web UI
3. Click on "Path Configuration" to set custom paths
4. Configure paths for:
   - Workspace directory
   - Evaluation directory
   - Python executable path
   - Conda environment path (optional)
   - Output directories
   - Log directories

Alternatively, you can manually edit `backend/path_config.json`:

```json
{
  "workspace_dir": "/path/to/reasoning-models-eval",
  "evaluation_dir": "/path/to/reasoning-models-eval/evaluation",
  "backend_dir": "/path/to/reasoning-models-eval/backend",
  "python_path": "/path/to/python",
  "conda_env_path": "/path/to/conda/env",
  "output_dir": "/path/to/outputs",
  "exports_dir": "/path/to/exports",
  "logs_dir": "/path/to/logs",
  "scripts_dir": "/path/to/scripts",
  "job_db_path": "/path/to/job_db.json",
  "openai_api_key": "",
  "hf_token": "",
  "slurm_partition": "gpu-a100-dev",
  "slurm_account": "YOUR_ACCOUNT",
  "slurm_wall_time": "1:00:00"
}
```

## Running the Application

### Quick Start

The easiest way to start the application is using the provided startup script:

```bash
# Make sure you're in the project root directory
cd reasoning-models-eval

# Make the script executable (if not already)
chmod +x start.sh

# Start the application
./start.sh
```

The startup script will:
1. Find an available port for the backend (starting from 8000)
2. Create a frontend configuration file with the backend URL
3. Start the FastAPI backend server
4. Start a simple HTTP server for the frontend
5. Display the URLs for both backend and frontend

### Manual Start

If you prefer to start the servers manually:

#### Start Backend

```bash
# Navigate to backend directory
cd backend

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Start Frontend

In a separate terminal:

```bash
# Navigate to frontend directory
cd frontend

# Create config.js with backend URL
cat > config.js << EOF
window.API_BASE = 'http://localhost:8000';
window.WS_BASE = 'ws://localhost:8000';
EOF

# Start HTTP server
python3 -m http.server 3000
```

### Access the Application

Once started, access the application at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Usage

### Job Configuration

1. **Add Model Configuration**: Click "+ Add Model Configuration" to create a new evaluation setup
2. **Model Selection**: 
   - Choose from predefined models (Qwen2.5-Math, etc.)
   - Or enter a custom Hugging Face model URL (e.g., `Qwen/Qwen2.5-Math-7B-Instruct`)
3. **Dataset Configuration**: 
   - Enter one dataset per line
   - Use built-in names (e.g., `gsm8k`, `math`, `gsm8k,math`)
   - Or use custom Hugging Face dataset URLs
4. **Prompt Configuration**:
   - Select a prompt type (cot, pal, tool-integrated, custom, etc.)
   - Click the preview icon to see the exact prompt with few-shot examples
   - For custom prompts, enter a template with `{question}` placeholder
5. **Hyperparameters**: Use multi-line text areas for batch evaluation:
   - **Temperature**: `0.0`, `0.1`, `0.5` (one per line)
   - **Top P**: `1.0`, `0.9`, `0.8` (one per line)
   - **Top K**: `0`, `10`, `50` (one per line)
   - **N Sampling**: `1`, `5`, `10` (one per line)
   - **Seeds**: `42`, `123`, `456` (one per line)
   - **Max Tokens**: `2048`, `4096` (one per line)
6. **Backend Selection**: 
   - **local**: Run on local machine (requires GPU)
   - **bash**: Run as background process
   - **slurm**: Submit to SLURM cluster
7. **Job Count**: The UI shows the total number of jobs to be created based on all combinations

### Job Management

- **Job List**: View all jobs with their current status (QUEUED, RUNNING, DONE, ERROR)
- **Job Monitoring**: Click "Monitor" to stream real-time logs via WebSocket
- **Job Cancellation**: Cancel running jobs with the "X" button in the monitoring view
- **Job Deletion**: Remove jobs from the list with the trash icon
- **Result Viewing**: Click "View Results" for completed jobs to access evaluation outputs

### Prompt Preview

The prompt preview feature allows you to see the exact prompt that will be sent to the model:

1. Select a prompt type from the dropdown
2. Click the preview icon (👁️) next to the prompt type
3. View the full prompt including:
   - Few-shot examples (if configured)
   - Prompt template formatting
   - Sample question demonstration

This helps you verify that your prompt configuration is correct before running the evaluation.

### SLURM Integration

For cluster execution:

1. **Configure SLURM Settings**: Set partition, account, and wall time in path configuration
2. **Submit Jobs**: Select "slurm" as the backend when creating jobs
3. **Monitor Jobs**: Jobs automatically update status every 5 seconds
4. **View SLURM Job IDs**: SLURM jobs display the actual SLURM job ID for reference
5. **Access Logs**: Log files are automatically named with SLURM job IDs

**SLURM Configuration**:
- **Partition**: `gpu-a100-dev` (default, customize as needed)
- **Account**: Your SLURM account name
- **Wall Time**: `1:00:00` (1 hour, default)

## API Endpoints

### Job Management

- `POST /jobs` - Create a new evaluation job
- `GET /jobs` - List all jobs
- `GET /jobs/{job_id}` - Get job status
- `POST /jobs/{job_id}/cancel` - Cancel a running job
- `DELETE /jobs/{job_id}` - Delete a job
- `GET /jobs/{job_id}/prob-file` - Get probability data file
- `GET /jobs/{job_id}/prob-plot` - Get probability plot

### Prompt Preview

- `POST /prompt/preview` - Get prompt preview with few-shot examples

### Configuration

- `GET /config/paths` - Get current path configuration
- `POST /config/paths` - Update path configuration
- `POST /config/paths/reset` - Reset to default configuration
- `GET /config/paths/validate` - Validate path configuration

### Real-time Monitoring

- `WS /stream/{job_id}` - WebSocket endpoint for live log streaming

### File Access

- `GET /file` - Access evaluation result files
- `GET /jobs/{job_id}/truncation-analysis/plot` - Get truncation analysis plots

## Troubleshooting

### Common Issues

#### 1. Missing Python Packages

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
# Make sure you're in the correct conda environment
conda activate reasoning-models-eval

# Install the missing package
pip install X
```

#### 2. Disk Quota Exceeded

**Error**: `No space left on device` or quota warnings

**Solution**:
- Clean up `~/.cache` directory
- Remove old conda environments
- Use `/work` for conda environments and logs
- Clean up old evaluation outputs

#### 3. Port Already in Use

**Error**: `Address already in use` when starting the backend

**Solution**:
- The startup script automatically finds an available port
- Or manually specify a different port:
  ```bash
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
  ```

#### 4. Backend Fails to Start

**Error**: Backend server fails to start

**Solution**:
- Check that all dependencies are installed
- Verify Python version (3.10+)
- Check backend logs for error messages
- Ensure ports are not blocked by firewall

#### 5. SLURM Job Submission Fails

**Error**: SLURM job submission errors

**Solution**:
- Verify SLURM account and partition settings
- Check that SLURM is available on your system
- Verify wall time limits
- Check SLURM error logs in `backend/logs/`

#### 6. Frontend Cannot Connect to Backend

**Error**: Frontend shows connection errors

**Solution**:
- Verify backend is running
- Check that `frontend/config.js` has the correct backend URL
- Ensure CORS is enabled (it should be by default)
- Check browser console for detailed error messages

#### 7. Job List Empty After Restart

**Solution**:
- Jobs are stored in `backend/job_db.json`
- If the file doesn't exist, jobs won't be restored
- This is expected for the first run

#### 8. WebSocket Closes Immediately

**Error**: WebSocket connection closes right after opening

**Solution**:
- For SLURM jobs, ensure the job is running and log file exists
- Check that the job hasn't completed
- Verify file permissions for log files

#### 9. Conda Environment Not Found

**Error**: Conda environment path doesn't exist

**Solution**:
- Verify conda environment path in `backend/path_config.json`
- Or leave `conda_env_path` as `null` to use system Python
- Ensure conda environment is activated when running

#### 10. GPU Not Available

**Error**: CUDA errors or GPU not found

**Solution**:
- Verify GPU is available: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- For SLURM, ensure GPU resources are requested
- For local execution, ensure GPU drivers are installed

### Getting Help

If you encounter issues not covered here:

1. Check the backend logs in `backend/logs/`
2. Check SLURM error logs in `backend/logs/` (for SLURM jobs)
3. Check browser console for frontend errors
4. Review the API documentation at `http://localhost:8000/docs`
5. Check GitHub issues for known problems

## Development

### Project Structure

The project follows a modular architecture:

- **Backend**: FastAPI application with async job management
- **Frontend**: Vanilla HTML/JavaScript (no build step required)
- **Evaluation**: Original Qwen2.5-Math evaluation code (unchanged)

### Key Constraint

**Re-use math_eval.py as-is; wrapper only.** The original Qwen2.5-Math evaluation code remains untouched. The API layer provides a web interface on top of the existing command-line tool.

### Adding New Features

1. **Backend**: Add new endpoints in `backend/app/main.py`
2. **Frontend**: Add UI components in `frontend/app.js` and `frontend/index.html`
3. **Schemas**: Define data models in `backend/app/schemas.py`
4. **Runner**: Modify job execution in `backend/app/runner.py`

### Testing

Run tests (if available):

```bash
# Navigate to backend directory
cd backend

# Run tests
python -m pytest tests/
```

## Configuration Reference

### Path Configuration

The path configuration file (`backend/path_config.json`) contains:

- **workspace_dir**: Root directory of the project
- **evaluation_dir**: Directory containing evaluation scripts
- **backend_dir**: Directory containing backend code
- **python_path**: Path to Python executable
- **conda_env_path**: Path to conda environment (optional)
- **output_dir**: Directory for evaluation outputs
- **exports_dir**: Directory for Excel exports
- **logs_dir**: Directory for log files
- **scripts_dir**: Directory for SLURM scripts
- **job_db_path**: Path to job database file
- **openai_api_key**: OpenAI API key (optional, for CoT analysis)
- **hf_token**: Hugging Face token (optional, for gated models)
- **slurm_partition**: SLURM partition name
- **slurm_account**: SLURM account name
- **slurm_wall_time**: SLURM wall time limit

### Environment Variables

The following environment variables can be used:

- **TOGETHER_API_KEY**: Together AI API key (for Together API)
- **HF_TOKEN**: Hugging Face token (for gated models)
- **OPENAI_API_KEY**: OpenAI API key (for CoT analysis)
- **CONDA_PREFIX**: Conda environment path (auto-detected)

## License

This project is based on the Qwen2.5-Math evaluation framework. Please refer to the original repository for license information.

## Acknowledgments

- **Qwen2.5-Math**: Original evaluation framework by Qwen Team
- **FastAPI**: Modern web framework for building APIs
- **Hugging Face**: Model and dataset hosting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the API documentation at `/docs`

---

**Last Updated**: November 2024
**Version**: 1.0.0
**Maintainer**: Reasoning Model Evaluator Team
