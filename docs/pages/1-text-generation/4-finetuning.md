---
title: Finetuning
parent: Text Generation
nav_order: 3
layout: page
permalink: /text-generation/finetuning/
---

## Text Generation Finetuning

HappyTextGeneration contains two methods for training 
- train(): fine-tune the model to understand a body of text better
- eval(): determine how well the model performs 

### train()

inputs: 
1. input_filepath (string): a path file to a text file that contains nothing but text to train the model.
2. args (GENTrainArgs): a dataclass with the same fields types as shown in Table 1.1. 


#### Table 1.1

| Parameter                     |Default|
|-------------------------------|-------|
| learning_rate                 | 5e-5  |
| num_train_epochs              | 3     |
| batch_size                    | 1     |
| weight_decay                  | 0     |
| adam_beta1                    | 0.9   |
| adam_beta2                    | 0.999 |
| adam_epsilon                  | 1e-8  |
| max_grad_norm                 | 1.0   |
| save_preprocessed_data        | False |
| save_preprocessed_data_path   | ""    |
| load_preprocessed_data        | False |
| load_preprocessed_data_path   | ""    |
| preprocessing_processes       | 1     |
| fp16                          | False |


Information about the learning parameters can be found [here](/learning-parameters/)

Information about saving/loading preprocessed data can be found [here](/save-load-data/)

preprocessing_processes: Number of processes used for preprocessing. We recommend 1-4. 



#### Example 1.3:
```python
from happytransformer import HappyGeneration, GENTrainArgs
# --------------------------------------#

happy_gen = HappyGeneration()
args = GENTrainArgs(num_train_epochs=1) 
happy_gen.train("../../data/gen/train-eval.txt", args=args)
```

### eval()
Input:
1. input_filepath (string): a path file to a text file with just text to evaluate  
2. args (WPEvalArgs): a dataclass with the same fields shown in Table  1.2

#### Table 1.2

| Parameter                     |Default|
|-------------------------------|-------|
| save_preprocessed_data        | False |
| save_preprocessed_data_path   | ""    |
| load_preprocessed_data        | False |
| load_preprocessed_data_path   | ""    |
| preprocessing_processes       | 1     |

See the explanations under Table 1.1 for more information 



Output: An object with the field "loss"

#### Example 1.4
```python
from happytransformer import HappyGeneration, GENEvalArgs
# --------------------------------------#
happy_gen = HappyGeneration()  
args = GENEvalArgs(preprocessing_processes=2)
result = happy_gen.eval("../../data/gen/train-eval.txt", args=args)
print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
print(result)  # EvalResult(loss=3.3437771797180176)
print(result.loss)  # 3.3437771797180176

```
