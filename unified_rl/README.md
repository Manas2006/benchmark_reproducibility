# Unified RL Framework

A unified interface for running reinforcement learning experiments across different backends (TRL, Open-R1, and Verl).

## Overview

This framework provides a consistent interface for running RL experiments while leveraging the unique capabilities of each backend library. It supports:

- Multiple RL algorithms (PPO, DPO, GRPO)
- Various model architectures
- Distributed training
- Flexible configuration system

## Installation

```bash
pip install -e .
```

## Quick Start

1. Choose your library (TRL, Open-R1, or Verl)
2. Configure your experiment in `configs/default.yaml`:
   - Set required fields (model, dataset, reward model)
   - Uncomment and modify library-specific settings if needed
3. Run training:

```bash
# Using TRL
python train.py library=trl

# Using Open-R1
python train.py library=openr1

# Using Verl
python train.py library=verl
```

### Minimal Configuration Example

```yaml
# Required fields
model:
  pretrained_name: "Qwen/Qwen2.5-Math-1.5B"
dataset:
  name: "your_dataset"
reward_model:
  type: "pipeline"
  name: "your_reward_model"

# Optional: Uncomment and modify library-specific settings
# library_specific:
#   trl:
#     optimizer:
#       gradient_accumulation_steps: 1
```

## Configuration System

The framework uses a hierarchical configuration system:

1. **Common Settings**: Basic parameters supported by all libraries
   - Model settings
   - Dataset settings
   - Basic optimizer and scheduler settings
   - Training parameters

2. **Library-Specific Settings**: Advanced features unique to each library
   - TRL: Additional optimizer and scheduler options
   - Verl: Distributed training, FSDP/Megatron backends
   - Open-R1: GRPO-specific settings

See `configs/default.yaml` for all available options.

## Library Support

### TRL

- Basic RL algorithms (PPO, DPO)
- Standard optimizer and scheduler support
- Good for single-GPU or small-scale training

### Open-R1

- Built on TRL
- Specialized in GRPO algorithm
- Inherits TRL's features plus GRPO-specific optimizations

### Verl

- Most extensive feature set
- Supports both FSDP and Megatron backends
- Advanced distributed training capabilities
- Rich optimizer and scheduler options

## Documentation

- [Optimizer and Scheduler Support](docs/optimizer_scheduler.md): Detailed comparison of optimizer and scheduler capabilities

## Project Structure

```
unified_rl/
├── configs/
│   └── default.yaml      # Main configuration file
├── algorithms/
│   ├── trl.py           # TRL adapter
│   ├── openr1.py        # Open-R1 adapter
│   └── verl.py          # Verl adapter
├── utils/
│   └── optim.py         # Optimizer utilities
├── docs/
│   └── optimizer_scheduler.md  # Optimizer documentation
└── train.py             # Main training script
```

## Workflow

1. **Configuration**:
   - Set common parameters in `default.yaml`
   - Add library-specific settings as needed
   - Configure model, dataset, and training parameters

2. **Training**:
   - The framework loads the configuration
   - Initializes the appropriate adapter (TRL/Open-R1/Verl)
   - Sets up the model, dataset, and reward function
   - Runs training with the specified settings

3. **Monitoring**:
   - Training progress is logged to the specified directory
   - Optional Weights & Biases integration
   - Checkpoints are saved periodically

## Best Practices

1. Start with common settings and add library-specific features as needed
2. Use the appropriate library for your scale:
   - TRL: Single-GPU or small-scale
   - Open-R1: GRPO-specific needs
   - Verl: Large-scale distributed training
3. Monitor training metrics and adjust configuration accordingly
4. Refer to library-specific documentation for advanced features
