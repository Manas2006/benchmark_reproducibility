# Reproducibility Dashboard

A complete full-stack dashboard for configuring and running hyperparameter sweeps with live monitoring and results analysis.

## Installation

```bash
# Install backend dependencies
cd backend && pip install -e .

# Install frontend dependencies
cd ../frontend && npm install
```

## Launch

```bash
# Launch the complete dashboard
reproducibility-cli webui
# or shorter alias
repro-cli webui
```

This will:

- Start the FastAPI backend on port 8000
- Start the React frontend on port 3000
- Open your browser to <http://localhost:3000>

## Workflow

1. **Configure Run**: Set up your experiment parameters including raw SBATCH directives, hyperparameters, and evaluation settings
2. **Generate & Run**: Click to start your experiment sweep and monitor progress in the Runs tab
3. **View Results**: Analyze results in the Results tab with sorting, filtering, and CSV export
4. **Unit Test**: Use the Unit Test feature to sanity-check one example before running the full sweep

## Integration with Testing Code

Place your existing `run.sh`, `evaluate.sh`, and `main.py` under `/testing` or reference them via `local_dir`.

Ensure your scripts accept the CLI flags described in the template and output standardized `RESULT: {...}` JSON lines for parsing.

## Features

- **Full Experiment Configuration**: SBATCH directives, hyperparameters, evaluation settings
- **Live Monitoring**: Real-time logs and job status tracking
- **Results Analysis**: Sortable, filterable results grid with CSV export
- **Unit Testing**: Quick validation of single examples
- **Flexible Evaluation**: Support for both rule-based and LLM-based evaluation
- **Horizontal Layout**: Full-width interface optimized for experiment management

## Project Structure

```
reproducibility-dashboard/
├── backend/                 # FastAPI backend with CLI
├── frontend/               # React frontend
├── tests/                  # Unit tests
└── README.md              # This file
```

## Development

### Backend

- FastAPI with CORS for localhost:3000
- Typer CLI for easy launching
- Jinja2 templates for script generation
- Subprocess management for SLURM jobs

### Frontend

- React with modern hooks
- Tailwind CSS for styling
- Real-time updates via polling
- Responsive design for experiment management

## License

MIT License - see LICENSE file for details.
