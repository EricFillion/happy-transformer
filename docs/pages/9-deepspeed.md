---
layout: page
title: Deepspeed Training
permalink: /deepspeed/
nav_order: 16
---

## Deepspeed Training

### Installation
Deepspeed is not installed by default and thus you must manually install it.
Here commands we recommend you try to install it within your local environment.

```shell
git clone https://github.com/microsoft/DeepSpeed.git

cd DeepSpeed

DS_BUILD_UTILS=1 pip install .

```
### TrainArgs

To use Deepspeed set your TrainArgs's deepspeed parameter to a path to a Deepspeed file as described [here](https://huggingface.co/docs/transformers/main_classes/deepspeed).


```python
from happytransformer import GENTrainArgs

args = GENTrainArgs(deepspeed="path-to-ds-file.json",
                    )
```

### Deepspeed File Example

```json
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": true
    },

    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 32,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}

```
