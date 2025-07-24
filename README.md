# Qwen Math Evaluation UI Pipeline

## Goal
Add a lightweight web UI to tweak hyper-parameters (temp/top-p/k, model, dataset, prompt, seed, eval method, n_sampling) and stream GPU/log progress for math_eval runs.

## Architecture
- **Backend**: FastAPI application that wraps the existing `math_eval.py`
- **Frontend**: Simple HTML/JavaScript UI for parameter configuration and real-time monitoring
- **Runner**: Async job management and progress streaming
- **Persistent Job DB**: Jobs are saved to disk and restored on backend restart

## Key Constraint
**Re-use math_eval.py as-is; wrapper only.** The original Qwen2.5-Math evaluation code remains untouched. Our API layer provides a web interface on top of the existing command-line tool.

## Directory Structure
```
qwen-eval-ui/
├── evaluation/           # Original Qwen2.5-Math repo (unchanged)
│   ├── math_eval.py     # Main evaluation script
│   └── ...
├── backend/             # FastAPI backend (new)
│   ├── app/
│   │   ├── main.py      # FastAPI entrypoint
│   │   ├── schemas.py   # Pydantic models
│   │   ├── runner.py    # Wraps math_eval.py
│   │   └── post_eval.py # Evaluation helpers
│   ├── requirements.txt # Backend dependencies
│   └── scripts/         # SLURM and run scripts
├── frontend/            # Simple HTML/JS UI (new)
│   ├── index.html       # Main UI page
│   └── app.js           # JavaScript functionality
├── logs/                # SLURM output/error logs
├── job_db.json          # Persistent job database
└── README_PIPELINE.md   # This file
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/QwenLM/Qwen2.5-Math qwen-eval-ui
cd qwen-eval-ui
```

### 2. Create a conda environment (in /work for disk space)
```bash
conda create -y -p /work/$USER/ls6/miniconda3/envs/qwen-eval python=3.10
conda activate /work/$USER/ls6/miniconda3/envs/qwen-eval
```

### 3. Install dependencies
- **Backend:**
```bash
pip install -r backend/requirements.txt
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
- Open the frontend in your browser.
- Configure evaluation parameters and submit jobs.
- Monitor jobs and tail logs in real time (enter backend UUID or SLURM job number).
- View job list and status persistently, even after backend restarts.

## SLURM/Cluster Notes
- SLURM jobs are submitted with Lonestar6-compatible batch scripts.
- Logs and scripts are written to `/work` for quota safety.
- If you see disk quota errors, clean up your home directory or use `/work` for environments and logs.
- If you see CUDA errors, ensure you are on a GPU node and CUDA is available.

## Troubleshooting
- **Missing Python packages:** Install them in your conda env.
- **Disk quota exceeded:** Clean up `~/.cache`, old venvs, or use `/work` for all environments and logs.
- **Job list empty after restart:** Only jobs submitted after persistence was enabled will appear. Old jobs may not be recoverable.
- **WebSocket closes immediately:** For SLURM jobs, make sure the job is running and the log file exists.
- **Other errors:** Check backend logs and SLURM `.err` files in `logs/`.

## Development Status
- ✅ Repository cloned
- ✅ Backend implemented with FastAPI
- ✅ Frontend implemented with simple HTML/JS
- ✅ Conda environment set up in /work
- ✅ Job management system with persistence
- ✅ Real-time monitoring via WebSocket
- ✅ Multiple model configuration support
- 🔄 Ready for testing and refinement 