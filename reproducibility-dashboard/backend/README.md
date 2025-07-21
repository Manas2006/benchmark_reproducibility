# Reproducibility Dashboard Backend

FastAPI backend for the Reproducibility Dashboard, providing REST API endpoints for managing hyperparameter sweeps and SLURM job execution.

## Features

- **SLURM Integration**: Submit and monitor SLURM jobs
- **Template Generation**: Jinja2 templates for experiment scripts
- **Job Management**: Track job status, logs, and results
- **Results Processing**: Parse and store experiment results
- **Unit Testing**: Single experiment validation
- **CLI Interface**: Typer-based command line interface

## Installation

```bash
pip install -e .
```

This installs the package in development mode with console script entry points.

## Development

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Run the CLI
reproducibility-cli webui
```

## Project Structure

```
app/
├── cli.py              # Typer CLI application
├── main.py             # FastAPI application
├── models.py           # Pydantic data models
├── job_manager.py      # SLURM job management
└── templates/
    └── run_script.j2   # Jinja2 template for SLURM scripts

tests/
└── test_unit_run.py    # Unit test for single experiments
```

## API Endpoints

### POST /run

Start a new experiment job.

**Request Body**: `RunRequest` model with experiment configuration

**Response**: `RunResponse` with run ID and experiment count

### GET /jobs

List all experiment jobs.

**Response**: Array of `JobSummary` objects

### GET /jobs/{run_id}/logs

Get logs for a specific job.

**Response**: `LogResponse` with logs and status

### DELETE /jobs/{run_id}

Cancel a running job.

**Response**: Success message

### GET /results

Get all experiment results.

**Response**: Array of result objects

### GET /results/download

Download results as CSV file.

**Response**: CSV file download

### POST /unit_test

Run a unit test with single experiment.

**Request Body**: `UnitTestRequest` model

**Response**: `UnitTestResponse` with test results

### GET /health

Health check endpoint.

**Response**: Status object

## Models

### RunRequest

Complete experiment configuration including:

- SBATCH directives
- Directory paths
- Model and dataset lists
- Hyperparameters (temperature, top_p, top_k, etc.)
- Evaluation settings
- Optional existing results path

### JobInfo

Internal job tracking with:

- Run ID and SLURM job ID
- Status and timing
- Parameters and logs
- Process management

### ExperimentResult

Result data structure with:

- Model and dataset information
- Hyperparameter values
- Metrics (accuracy, loss, runtime)
- Evaluation settings
- Timestamp

## Job Management

The `JobManager` class handles:

- SLURM job submission and monitoring
- Log collection and parsing
- Result storage in CSV format
- Job status tracking

## Template System

Uses Jinja2 templates to generate SLURM scripts that:

- Include raw SBATCH directives
- Set up environment (conda)
- Run experiments with proper parameters
- Handle both generation and evaluation modes
- Support existing results evaluation

## CLI Commands

### webui

Launch the complete dashboard (backend + frontend + browser)

### install

Install all dependencies for the project

### test

Run the test suite

## Integration

The backend expects your experiment scripts (`run.sh`, `evaluate.sh`) to:

- Accept the CLI parameters defined in the template
- Output `RESULT: {...}` JSON lines for result parsing
- Handle the evaluation settings passed as arguments

## Development Notes

- Uses subprocess for SLURM interaction
- Threading for job monitoring
- CSV storage for results
- JSON parsing for RESULT lines
- Comprehensive error handling
- Health checks and status monitoring
