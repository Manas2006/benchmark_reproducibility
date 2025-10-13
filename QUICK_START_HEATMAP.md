# Quick Start: Heatmap Visualization

## 🚀 Get Started in 3 Steps

### 1. Run an Evaluation with Probability Tracking

```bash
# Make sure to enable probability tracking
python evaluation/math_eval.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --dataset gsm8k \
    --enable_prob_tracking \
    --backend slurm
```

### 2. Start the Application

```bash
cd qwen-eval-ui
./start.sh
```

### 3. View Heatmaps

1. Open http://localhost:3000
2. Click **"Heatmap Visualization"** tab
3. Select your completed job
4. Choose a question
5. Explore the interactive heatmaps! 🎨

## What You'll See

- **Left Heatmap**: Model's confidence in tokens it chose
- **Right Heatmap**: Probability of correct answer tokens
- **Hover**: See exact probability percentages
- **Colors**: White = low confidence, Red = high confidence

## Troubleshooting

- **No jobs available?** Make sure `enable_prob_tracking: true` was used
- **Backend errors?** Check that port 8000 is free
- **All white tokens?** Normal for correct probabilities (they're very small)

## Need Help?

See `HEATMAP_FEATURE.md` for detailed documentation.
