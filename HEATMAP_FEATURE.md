# Heatmap Visualization Feature

## Overview

This feature adds **Token Probability Heatmaps** to the Qwen Math Evaluation UI, allowing you to visualize the model's confidence at each token level during mathematical reasoning.

## What It Shows

The heatmap displays two side-by-side visualizations:

1. **Chosen Token Probabilities** (Left): Shows the model's confidence in the tokens it actually chose
2. **Correct Token Probabilities** (Right): Shows the probability of the correct answer token at each step (log-scaled for better visualization)

## How to Use

### 1. Start the Application

```bash
cd qwen-eval-ui
./start.sh
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000

### 2. Access the Heatmap Feature

1. Open your browser and go to http://localhost:3000
2. Click on the **"Heatmap Visualization"** tab
3. Select a completed job with probability tracking from the dropdown
4. Choose a question from that job
5. View the interactive heatmaps!

### 3. Understanding the Visualization

- **Color Intensity**: White = low confidence, Red = high confidence
- **Hover Tooltips**: Hover over any token to see its exact probability percentage
- **Token-by-Token Analysis**: Each word/token is highlighted based on its probability

## Technical Details

### Backend Changes

- **New API Endpoints**:
  - `GET /jobs/{job_id}/questions` - Lists available questions for a job
  - `GET /jobs/{job_id}/heatmap-data/{question_idx}` - Returns token probability data

- **Data Processing**:
  - Uses `chosen_token_probs` for model's chosen token confidence
  - Uses `probability_log` for correct answer token probabilities
  - Applies log-scaling to make small probabilities visible
  - Handles different output formats (list vs string)

### Frontend Changes

- **New UI Tab**: "Heatmap Visualization" with dropdowns for job/question selection
- **Interactive Heatmaps**: Real-time rendering with hover tooltips
- **Color Gradient**: White-to-red gradient based on probability values
- **Responsive Design**: Side-by-side layout with proper spacing

### Key Files Modified

- `backend/app/main.py` - New API endpoints and data processing
- `backend/app/schemas.py` - New Pydantic models for API responses
- `frontend/index.html` - New heatmap UI components
- `frontend/app.js` - Heatmap rendering and interaction logic

## Requirements

- Completed evaluation jobs with `enable_prob_tracking: true`
- Probability data files (`.jsonl` files with `_prob` suffix)
- Python 3.10+ with required dependencies

## Troubleshooting

### Common Issues

1. **"No jobs with probability tracking"**: Ensure your evaluation jobs have `enable_prob_tracking: true`
2. **"Failed to fetch"**: Check that the backend is running on port 8000
3. **All tokens appear white**: This is normal for correct token probabilities - they're typically very small and log-scaled

### Memory Issues

If you encounter memory errors:
- The system includes automatic GPU memory cleanup
- vLLM models are properly unloaded before HuggingFace model loading
- Garbage collection is forced between model switches

## Example Use Cases

1. **Debugging Model Reasoning**: See where the model was most/least confident
2. **Error Analysis**: Identify which tokens led to incorrect answers
3. **Confidence Analysis**: Compare chosen vs correct token probabilities
4. **Research**: Study model behavior during mathematical problem-solving

## Data Format

The heatmap expects probability data in this format:
```json
{
  "chosen_token_probs": {"epoch_0": [0.8, 0.9, 0.7, ...]},
  "probability_log": {"epoch_0": [1e-6, 1e-8, 1e-5, ...]},
  "entropies": {"epoch_0": [0.2, 0.1, 0.3, ...]}
}
```

## Future Enhancements

- Support for different tokenization methods
- Batch processing for multiple questions
- Export functionality for heatmap data
- Additional visualization modes (entropy, uncertainty)

---

**Note**: This feature requires probability tracking to be enabled during evaluation. Make sure to set `enable_prob_tracking: true` when running evaluations.
