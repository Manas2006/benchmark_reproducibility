# Qwen Math Evaluation UI Pipeline

## Goal
Add a lightweight web UI to tweak hyper-parameters (temp/top-p/k, model, dataset, prompt, seed, eval method, n_sampling, max_tokens) and stream GPU/log progress for math_eval runs. Support both local execution and SLURM cluster jobs with comprehensive job management.

## Architecture
- **Backend**: FastAPI application that wraps the existing `math_eval.py`
- **Frontend**: Simple HTML/JavaScript UI for parameter configuration and real-time monitoring
- **Runner**: Async job management and progress streaming with UUID-based tracking
- **Persistent Job DB**: Jobs are saved to disk and restored on backend restart
- **SLURM Integration**: Full SLURM job management with automatic status polling and log streaming

## Key Features
- **Multi-Parameter Support**: Configure temperature, top-p, top-k, n_sampling, seeds, max_tokens with multiple values per line for batch evaluation
- **Model Flexibility**: Use predefined models or custom Hugging Face model URLs
- **Dataset Flexibility**: Use built-in datasets or custom Hugging Face dataset URLs
- **Real-time Monitoring**: Live log streaming with WebSocket support for both local and SLURM jobs
- **Job Management**: View, cancel, delete, and monitor jobs with persistent storage
- **Result Viewing**: Direct access to evaluation results for completed jobs
- **SLURM Integration**: Automatic job status updates, log file management, and SLURM job ID display

## Key Constraint
**Re-use math_eval.py as-is; wrapper only.** The original Qwen2.5-Math evaluation code remains untouched. Our API layer provides a web interface on top of the existing command-line tool.

## Directory Structure
```
reasoning-models-eval/
├── evaluation/           # Original Qwen2.5-Math repo (unchanged)
│   ├── math_eval.py     # Main evaluation script
│   └── ...
├── backend/             # FastAPI backend (new)
│   ├── app/
│   │   ├── main.py      # FastAPI entrypoint with job management APIs
│   │   ├── schemas.py   # Pydantic models for requests/responses
│   │   ├── runner.py    # Job execution wrapper for math_eval.py
│   │   └── enums.py     # Backend type definitions
│   ├── job_db.json      # Persistent job database
│   └── scripts/         # SLURM and run scripts
├── frontend/            # Simple HTML/JS UI (new)
│   ├── index.html       # Main UI page with tabs
│   └── app.js           # JavaScript functionality
├── logs/                # SLURM output/error logs
└── output/              # Evaluation results (JSONL files)
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/QwenLM/Qwen2.5-Math reasoning-models-eval
cd reasoning-models-eval
```

### 2. Create a conda environment (in /work for disk space)
```bash
conda create -y -p /work/$USER/ls6/miniconda3/envs/qwen-eval python=3.10
conda activate /work/$USER/ls6/miniconda3/envs/qwen-eval
```

### 3. Install dependencies
- **Backend:**
```bash
pip install fastapi uvicorn pydantic
```
- **Evaluation engine:**
```bash
cd evaluation
pip install -r requirements.txt
pip install latex2sympy2 multiprocess datasets vllm tqdm torch transformers python-dateutil flash-attn sympy==1.12 antlr4-python3-runtime==4.11.1 word2number Pebble timeout-decorator
cd ..
```
- If `flash-attn` fails, ignore unless you need it for speed.

### 4. Start the backend and frontend
```bash
./start.sh
```
- Backend: http://localhost:8000
- Frontend: http://localhost:3001

## Usage

### Job Configuration
1. **Add Model Configuration**: Click "+ Add Model Configuration" to create evaluation setups
2. **Model Selection**: Choose from predefined models or enter a custom Hugging Face model URL
3. **Dataset Configuration**: Enter one dataset per line (built-in names or Hugging Face URLs)
4. **Hyperparameters**: Use multi-line text areas for batch evaluation:
   - Temperature: `0.0`, `0.1`, `0.5` (one per line)
   - Top P: `1.0`, `0.9`, `0.8` (one per line)
   - Top K: `0`, `10`, `50` (one per line)
   - N Sampling: `1`, `5`, `10` (one per line)
   - Seeds: `42`, `123`, `456` (one per line)
   - Max Tokens: `2048`, `4096` (one per line)
5. **Backend Selection**: Choose `local`, `bash`, or `slurm` execution
6. **Job Count**: The UI shows total jobs to be created based on all combinations

### Job Management
- **Job List**: View all jobs with status (QUEUED, RUNNING, DONE, ERROR)
- **Job Monitoring**: Click "Monitor" to stream real-time logs via WebSocket
- **Job Cancellation**: Cancel running jobs with the "X" button in monitoring
- **Job Deletion**: Remove jobs from the list with the trash icon
- **Result Viewing**: Click "View Results" for completed jobs to access evaluation outputs

### SLURM Integration
- **Automatic Status Updates**: Job status automatically updates every 5 seconds
- **SLURM Job ID Display**: SLURM jobs show the actual SLURM job ID for easy reference
- **Log Management**: Log files are automatically named with SLURM job IDs
- **Error Handling**: Comprehensive error reporting for SLURM submission issues

## API Endpoints

### Job Management
- `POST /jobs` - Create a new evaluation job
- `GET /jobs` - List all jobs
- `GET /jobs/{job_id}` - Get job status (uses UUID)
- `POST /jobs/{job_id}/cancel` - Cancel a running job
- `DELETE /jobs/{job_id}` - Delete a job

### Real-time Monitoring
- `WS /stream/{job_id}` - WebSocket endpoint for live log streaming

## SLURM/Cluster Notes
- SLURM jobs use the `gpu-a100-dev` partition with 1-hour wall time limit
- Logs and scripts are written to `/work` for quota safety
- Job tracking uses UUIDs internally, but displays SLURM job IDs for user convenience
- Automatic status polling ensures UI stays synchronized with SLURM queue
- If you see disk quota errors, clean up your home directory or use `/work` for environments and logs

## Troubleshooting
- **Missing Python packages:** Install them in your conda env using `conda install`
- **Disk quota exceeded:** Clean up `~/.cache`, old venvs, or use `/work` for all environments and logs
- **Job list empty after restart:** Only jobs submitted after persistence was enabled will appear
- **WebSocket closes immediately:** For SLURM jobs, make sure the job is running and the log file exists
- **SLURM submission errors:** Check wall time limits and partition availability
- **Other errors:** Check backend logs and SLURM `.err` files in `logs/`

## Development Status
- ✅ Repository cloned and refactored
- ✅ Backend implemented with FastAPI and UUID-based job tracking
- ✅ Frontend implemented with comprehensive job management UI
- ✅ Conda environment set up in /work
- ✅ Job management system with persistence and SLURM integration
- ✅ Real-time monitoring via WebSocket for local and SLURM jobs
- ✅ Multi-parameter batch evaluation support
- ✅ Result file viewing and job status management
- ✅ SLURM job ID display and automatic status updates
- 🔄 Ready for production use and further enhancements 