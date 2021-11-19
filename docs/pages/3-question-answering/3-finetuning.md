---
title: Finetuning
parent: Question Answering
nav_order: 3
layout: page
permalink: /question-answering/finetuning/
---

## Question Answering Finetuning

HappyQuestionAnswering contains three methods for training 
- train(): fine-tune a question answering model to become better at a certain task
- eval(): determine how well the model performs on a labeled dataset
- test(): run the model on an unlabeled dataset to produce predictions  

### train()

inputs: 
1. input_filepath (string): a path file to a csv file as described in table 3.0
2. args (QATrainArgs): a dataclass of type QATrainArgs which contains the fields shown in table 3.1


#### Table 3.0

1. context (string): background information for answer the question
2. question (string): the question that will be asked 
3. answer_text(string): the answer in string format 
4. answer_start(int): the char index of the start of the answer

| context                   | question          | answer_text   | answer_start |
|---------------------------|-------------------|---------------|--------------|
| October 31st is the date  | what is the date? | October 31st  | 0            |
| The date is November 23rd | what is the date? | November 23rd | 12           |



Information about the learning parameters can be found [here](/learning-parameters/)

Information about saving/loading preprocessed data can be found [here](/save-load/)

#### Table 3.1

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
| fp16                          | False |

Output: None
 

#### Example 3.3:
```python
from happytransformer import HappyQuestionAnswering, QATrainArgs
# --------------------------------------#
happy_qa = HappyQuestionAnswering()
args = QATrainArgs(num_train_epochs=1)
happy_qa.train("../../data/qa/train-eval.csv", args=args)

```

### eval()
Input:
1. input_filepath (string): a path file to a csv file as described in Table 3.0
2. args (QAEvalArgs): a dataclass of type QAEvalArgs which contains the fields shown in Table 3.3


Information about saving/loading preprocessed data can be found [here](/save-load-data/)

#### Table 3.2

| Parameter                     |Default|
|-------------------------------|-------|
| save_preprocessed_data        | False |
| save_preprocessed_data_path   | ""    |
| load_preprocessed_data        | False |
| load_preprocessed_data_path   | ""    |



Output: A dataclass with the variable "loss"

#### Example 3.4:
```python
from happytransformer import HappyQuestionAnswering, QAEvalArgs
# --------------------------------------#
happy_qa = HappyQuestionAnswering()
args = QAEvalArgs() #  The default settings as an example
result = happy_qa.eval("../../data/qa/train-eval.csv")
print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
print(result)  # EvalResult(eval_loss=0.11738169193267822)
print(result.loss)  # 0.1173816919326782

```

### test()
Input:
1. input_filepath (string): a path file to a csv file as described in table 3.2
2. args (QATestArgs): A dataclass that contains the values shown in table x above

#### Table 3.3

1. context (string): background information for answer the question
2. question (string): the question that will be asked 

| context                   | question          | 
|---------------------------|-------------------|
| October 31st is the date  | what is the date? |
| The date is November 23rd | what is the date? | 


Output: A list of named tuples with keys: "answer", "score", "start" and "end"


#### Example 3.5:
```python
from happytransformer import HappyQuestionAnswering, QATestArgs
# --------------------------------------#
happy_qa = HappyQuestionAnswering()
args = QATestArgs() #  Using the default settings as an example
result = happy_qa.test("../../data/qa/test.csv", args=args)
print(type(result))
print(result)  # [QuestionAnsweringResult(answer='October 31st', score=0.9939756989479065, start=0, end=12), QuestionAnsweringResult(answer='November 23rd', score=0.967872679233551, start=12, end=25)]
print(result[0])  # QuestionAnsweringResult(answer='October 31st', score=0.9939756989479065, start=0, end=12)
print(result[0].answer)  # October 31st

```

