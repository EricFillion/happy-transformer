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
2. args (WPTrainArgs): a dataclass with the same fields types as shown in table 4.1. 
3. eval_filepath (string): By default, an evaluating dataset will be generated from the supplied training data. But, you may provide a filepath to a text of CSV file as described for input_filepath to use standalone evaluating data. 

#### Table 4.0
| text                           |
|--------------------------------|
| This is a training case.       | 
| This is another training case  | 

#### Table 4.1

| Parameter               | Default             |
|-------------------------|---------------------|
| learning_rate           | 5e-5                |
| num_train_epochs        | 1                   |
| batch_size              | 1                   |
| weight_decay            | 0                   |
| save_path               | ""                  |
| load_path               | ""                  |
| preprocessing_processes | 1                   |
| mlm_probability         | 0.15                |
| line-by-line            | False               |
| fp16                    | False               |
| eval_ratio              | 0.1                 |
| save_steps              | 0.0                 |
| eval_steps              | 0.1                 |
| logging_steps           | 0.1                 |
| output_dir              | "happy_transformer" |
| padding                 | "max_length"        |
| truncation              | True                |
| max_length              | None                |

Information about the learning parameters can be found [here](/learning-parameters/)

Information about saving/loading preprocessed data can be found [here](/save-load/)

preprocessing_processes: Number of processes used for preprocessing. We recommend 1-4. 

mlm_probability: The probability of masking a token.

line-by-line: If False, training data is concatenated and then divided into sections that are the length of the model's input size, other than the last input which may be shorter. 
              If True, each input contains the text from a single line within the training data. The text may be truncated if the line is too long (eg BERT's max input size is 512 tokens). 

padding/truncation/max_length (experimental): See this [table](https://huggingface.co/docs/transformers/pad_truncation) to learn more.  We may modify or remove these parameters in future releases.

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
2. args (WPEvalArgs): a dataclass with the fields shown in Table 4.2
 
#### Table 4.2

| Parameter               | Default      |
|-------------------------|--------------|
| save_path               | ""           |
| load_path               | ""           |
| line-by-line            | False        |
| padding                 | "max_length" |
| truncation              | True         |
| max_length              | None         |

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

