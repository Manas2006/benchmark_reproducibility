# Optimizer and Scheduler Support

This document outlines the optimizer and scheduler capabilities supported by each library in the unified RL framework.

## Common Features

All libraries support these basic optimizer and scheduler features:

### Optimizers

- AdamW (default)
- Adam
- SGD
- Lion

Common parameters:

- Learning rate
- Weight decay
- Beta parameters (for Adam-based optimizers)
- Epsilon

### Schedulers

- Linear
- Cosine
- Constant
- Constant with warmup

Common parameters:

- Warmup ratio
- Total steps

## Library-Specific Features

### TRL

TRL provides a straightforward implementation with good support for common optimizers and schedulers.

#### Optimizer Features

- All common optimizers
- Gradient accumulation
- Gradient clipping
- Mixed precision training

#### Scheduler Features

- All common schedulers
- Additional cosine scheduler parameters:
  - Number of cycles
  - Minimum learning rate
- Additional warmup parameters for constant scheduler

### Verl

Verl offers the most extensive optimizer and scheduler support, with both FSDP and Megatron backends.

#### Optimizer Features

- All common optimizers
- Advanced gradient clipping
- Learning rate warmup with initialization
- Weight decay scheduling
- Checkpoint-based scheduler state
- Distributed optimizer support

#### Scheduler Features

- All common schedulers
- Additional scheduler types:
  - Inverse square root
  - Warmup-Stable-Decay (WSD)
- Advanced scheduling features:
  - Learning rate decay styles
  - Weight decay increment styles
  - WSD decay styles
  - Checkpoint-based scheduler state

#### Backend-Specific Features

- FSDP backend:
  - Full sharding support
  - Mixed precision training
  - CPU offloading
- Megatron backend:
  - Pipeline parallelism
  - Tensor parallelism
  - Expert parallelism

### Open-R1

Open-R1 is built on top of TRL and inherits its optimizer and scheduler support, with additional GRPO-specific features.

#### Optimizer Features

- Inherits all TRL optimizer features
- No additional optimizer-specific features

#### Scheduler Features

- Inherits all TRL scheduler features
- No additional scheduler-specific features

#### GRPO-Specific Features

- Score scaling
- Score normalization
- Score clipping

## Usage Examples

### Basic Usage (All Libraries)

```yaml
optimizer:
  type: AdamW
  args:
    lr: 3e-5
    weight_decay: 0.01

scheduler:
  type: linear
  args:
    warmup_ratio: 0.1
    total_steps: 100000
```

### Advanced Usage (Verl)

```yaml
library_specific:
  verl:
    backend: fsdp
    optimizer:
      clip_grad: 1.0
      lr_warmup_init: 0.0
      lr_decay_style: cosine
      weight_decay_incr_style: linear
    distributed:
      nnodes: 2
      n_gpus_per_node: 8
```

### GRPO Usage (Open-R1)

```yaml
library_specific:
  openr1:
    grpo:
      use_score_scaling: true
      use_score_norm: true
      score_clip: 0.5
```

## Best Practices

1. **Start Simple**: Begin with common settings and add library-specific features as needed.
2. **Check Compatibility**: Verify that your chosen optimizer/scheduler combination is supported by your library.
3. **Monitor Training**: Watch for signs of training instability that might indicate optimizer/scheduler issues.
4. **Use Library-Specific Features**: Take advantage of advanced features when available, especially for Verl's distributed training capabilities.
5. **Documentation**: Always refer to the library's documentation for the most up-to-date feature support.
