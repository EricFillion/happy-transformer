---
title: Finetuning
parent: Text-to-Text
nav_order: 3
layout: page
permalink: /text-to-text/finetuning/
---

## Text Generation Finetuning

TextGeneration contains two methods for training 
- train(): fine-tune the model to understand a body of text better
- eval(): determine how well the model performs 

### train()

inputs: 
1. input_filepath (string): a path file to a csv file that contains nothing but text to train the model.
2. args (TTTrainArgs): a dataclass with the same fields types as shown in Table 1.1. 

todo provide a sample CSV file 

#### Table 7.1

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
| max_input_length              | None  |
| max_output_length             | None  |


Information about the learning parameters can be found [here](/learning-parameters/)

Information about saving/loading preprocessed data can be found [here](/save-load-data/)

preprocessing_processes: Number of processes used for preprocessing. We recommend 1-4. 
max_input_length: The maximum number of tokens for the input. The rest get truncated. If None, uses the maximum number of tokens as permitted by the tokenizer. 
max_output_length: Ditto, except for the output. 



#### Example 7.3:
```python
    from happytransformer import HappyTextToText, TTTrainArgs
    # --------------------------------------#
    
    happy_tt = HappyTextToText()
    args = TTTrainArgs(num_train_epochs=1) 
    happy_tt.train("../../data/tt/train.txt", args=args)
```

### eval()
Input:
1. input_filepath (string): a path file to a csv file with the same format as described for the training data
2. args (TTEvalArgs): a dataclass with the same fields shown in Table  7.2

#### Table 7.2

| Parameter                     |Default|
|-------------------------------|-------|
| save_preprocessed_data        | False |
| save_preprocessed_data_path   | ""    |
| load_preprocessed_data        | False |
| load_preprocessed_data_path   | ""    |
| preprocessing_processes       | 1     |
| max_input_length              | None  |
| max_output_length             | None  |

See the explanations under Table 7.1 for more information 


Output: An object with the field "loss"

#### Example 1.4
```python
 

```
