---
title: Finetuning
parent: Word Prediction
nav_order: 2
layout: page
permalink: /word-prediction/finetuning/
---

## Word Prediction Finetuning

HappyWordPrediction contains two methods for training 
- train(): fine-tune the model to understand a body of text better
- eval(): determine how well the model performs 

### train()

inputs: 
1. input_filepath (string): a path file to a text file that contains nothing but text to train the model with
2. args (WPTrainArgs): a dataclass with the same fields types as shown in table 3.0. 


#### Table 4.0

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
| mlm_probability               | 0.15  |
| line-by-line                  | False |
| fp16                          | False |


Information about the learning parameters can be found [here](/learning-parameters/)

Information about saving/loading preprocessed data can be found [here](/save-load/)

preprocessing_processes: Number of processes used for preprocessing. We recommend 1-4. 

mlm_probability: The probability of masking a token.

line-by-line: If False, training data is concatenated and then divided into sections that are the length of the model's input size, other than the last input which may be shorter. 
              If True, each input contains the text from a single line within the training data. The text may be truncated if the line is too long (eg BERT's max input size is 512 tokens). 


#### Example 4.4:
```python
from happytransformer import HappyWordPrediction, WPTrainArgs
# --------------------------------------#
happy_wp = HappyWordPrediction()
args = WPTrainArgs(num_train_epochs=1) 
happy_wp.train("../../data/wp/train-eval.txt", args=args)
```

### eval()
Input:
1. input_filepath (string): a path file to text file with just text to evaluate 
2. args (WPEvalArgs): a dataclass with the fields shown in Table 4.1
 
#### Table 4.1

| Parameter                     |Default|
|-------------------------------|-------|
| save_preprocessed_data        | False |
| save_preprocessed_data_path   | ""    |
| load_preprocessed_data        | False |
| load_preprocessed_data_path   | ""    |
| preprocessing_processes       | 1     |
| line-by-line                  | False |

See the explanations under Table 4.0 for more information 

Output: An object with the field "loss"

#### Example 4.5
```python
from happytransformer import HappyWordPrediction, WPEvalArgs
# --------------------------------------#
happy_wp = HappyWordPrediction()  
args = WPEvalArgs(preprocessing_processes=2)
result = happy_wp.eval("../../data/wp/train-eval.txt", args=args)
print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
print(result)  # EvalResult(eval_loss=0.459536075592041)
print(result.loss)  # 0.459536075592041
```

