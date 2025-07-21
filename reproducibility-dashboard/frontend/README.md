# Reproducibility Dashboard Frontend

React frontend for the Reproducibility Dashboard, providing a modern web interface for configuring and monitoring hyperparameter sweeps.

## Features

- **Full Experiment Configuration**: SBATCH directives, hyperparameters, evaluation settings
- **Live Monitoring**: Real-time logs and job status tracking
- **Results Analysis**: Sortable, filterable results grid with CSV export
- **Unit Testing**: Quick validation of single examples
- **Horizontal Layout**: Full-width interface optimized for experiment management

## Installation

```bash
npm install
```

## Development

```bash
npm start
```

This will start the development server on <http://localhost:3000>.

## Build

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Project Structure

```
src/
├── components/           # React components
│   ├── RawSbatchInput.jsx
│   ├── ListInput.jsx
│   ├── SettingsInput.jsx
│   ├── GenerationConfig.jsx
│   ├── EvaluationConfig.jsx
│   ├── RunTable.jsx
│   ├── ResultsGrid.jsx
│   └── UnitTestButton.jsx
├── App.jsx              # Main application component
├── api.js               # API client functions
├── index.js             # React entry point
└── index.css            # Global styles with Tailwind
```

## Components

### RawSbatchInput

Textarea for entering raw SBATCH directives.

### ListInput

Multi-line input for lists of values (models, datasets, hyperparameters).

### SettingsInput

Single-value input with validation and helper text.

### GenerationConfig

Configuration section for generation parameters (models, datasets, hyperparameters).

### EvaluationConfig

Configuration section for evaluation settings (metrics, tools, prompts).

### RunTable

Table displaying active and completed experiment runs with live logs.

### ResultsGrid

Sortable and filterable results table with CSV export.

### UnitTestButton

Button for running single experiment tests to validate configuration.

## API Integration

The frontend communicates with the FastAPI backend through the `api.js` module, which provides functions for:

- Running experiments
- Fetching job status and logs
- Downloading results
- Running unit tests

## Styling

Uses Tailwind CSS for responsive, modern styling with a clean, professional appearance optimized for experiment management workflows.

## Development Notes

- Real-time updates via polling (every 2 seconds for jobs, logs)
- Form validation with error display
- Responsive design for different screen sizes
- Horizontal layout optimized for wide screens
- Comprehensive error handling and loading states
