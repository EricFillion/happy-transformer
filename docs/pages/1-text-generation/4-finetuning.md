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
1. input_filepath (string): a path file to a text file that contains nothing but text to train the model or a CSV file with a single column called text as shown in Table 1.0 .
2. args (GENTrainArgs): a dataclass with the same fields types as shown in Table 1.1. 
3. eval_filepath (string): By default, an evaluating dataset will be generated from the supplied training data. But, you may provide a filepath to a text of CSV file as described for input_filepath to use standalone evaluating data. 

#### Table 1.0

| text                           |
|--------------------------------|
| This is a training case.       | 
| This is another training case  | 

#### Table 1.1

| Parameter                   | Default             |
|-----------------------------|---------------------|
| learning_rate               | 5e-5                |
| num_train_epochs            | 1                   |
| batch_size                  | 1                   |
| weight_decay                | 0                   |
| save_path                   | ""                  |
| load_path                   | ""                  |
| fp16                        | False               |
| eval_ratio                  | 0.1                 |
| save_steps                  | 0.0                 |
| eval_steps                  | 0.1                 |
| logging_steps               | 0.1                 |
| output_dir                  | "happy_transformer" |
| max_length                  | None                |


Information about the learning parameters can be found [here](/learning-parameters/)

Information about saving/loading preprocessed data can be found [here](/save-load-data/)


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

| Parameter               | Default      |
|-------------------------|--------------|
| save_path               | ""           |
| load_path               | ""           |
| max_length              | None         |

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
