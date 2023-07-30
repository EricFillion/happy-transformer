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

## Arguments
To use Deepspeed set your TrainArgs's or EvalArg's __deepspeed__ parameter to a path to a Deepspeed file as described [here](https://huggingface.co/docs/transformers/main_classes/deepspeed). Below are options for what you may supply to the __deepspeed__ parameter for training and evaluating.

 TrainArgs:

| Value           | Type | Meaning                                                                                            |
|-----------------|------|----------------------------------------------------------------------------------------------------|
| False (default) | bool | DeepSpeed will not be used.                                                                        |
| "ZERO-2"        | str  | ZERO-2 is used.                                                                               |
| "ZERO-3"        | str  | ZERO-3 is used.                                                                               |
| "path-to-json"  | str  | You may provide a path to a JSON file with the format as described [here](https://huggingface.co/docs/transformers/main_classes/deepspeed) to use custom settings |

EvalArgs:
ZERO-2 is not compatible with evaluating.

| Value           | Type | Meaning                                                                                            |
|-----------------|------|----------------------------------------------------------------------------------------------------|
| False (default) | bool | DeepSpeed will not be used.                                                                        |
| "ZERO-3"        | str  | ZERO-3 is used.                                                                               |
| "path-to-json"  | str  | You may provide a path to a JSON file with the format as described [here](https://huggingface.co/docs/transformers/main_classes/deepspeed) to use custom settings |


```python
from happytransformer import GENTrainArgs, GENEvalArgs

train_args = GENTrainArgs(deepspeed="ZERO-3")

eval_args = GENEvalArgs(deepspeed="ZERO-3")

```

## Script

You __MUST__ run the code from a script for Deeppeed to work as intended. Use the command "deepspeed" instead of "python3" to run the script. Supply the flag "num_gpus" to specify the number of Nvidia GPUs you would like to use.

```python

deepspeed --num_gpus=2 train.py

```


