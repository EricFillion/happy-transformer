---
title: Finetuning
parent: Token Classification
nav_order: 2
layout: page
permalink: /token-classification/finetuning/
---


## Question Answering Finetuning

HappyQuestionAnswering contains three methods for training 
- train(): fine-tune a question answering model  to become better at a certain task
- eval(): determine how well the model performs on a labeled dataset
- test(): run the model on an unlabeled dataset to produce predictions  

### train()

inputs: 
1. input_filepath (string): a path file to a csv file as described in table 3.1
2. args (dictionary): a dictionary with the same keys and value types as shown below. 
The dictionary below shows the default values. 

Information about what the keys mean can be accessed [here](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)
```python

ARGS_QA_TRAIN= {
    'learning_rate': 5e-5,
    'weight_decay': 0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'max_grad_norm':  1.0,
    'num_train_epochs': 3.0,

}
```
Output: None
 
#### Table 3.1

1. context (string): background information for answer the question
2. question (string): the question that will be asked 
3. answer_text(string): the answer in string format 
4. answer_start(int): the char index of the start of the answer

| context                   | question          | answer_text   | answer_start |
|---------------------------|-------------------|---------------|--------------|
| October 31st is the date  | what is the date? | October 31st  | 0            |
| The date is November 23rd | what is the date? | November 23rd | 12           |

#### Example 3.3:
```python
    from happytransformer import HappyQuestionAnswering
    # --------------------------------------#
    happy_qa = HappyQuestionAnswering()
    happy_qa.train("../../data/qa/train-eval.csv")

```

### eval()
Input:
1. input_filepath (string): a path file to a csv file as described in table 3.1

output:

A dataclass with the variable "loss"

#### Example 3.4:
```python
    from happytransformer import HappyQuestionAnswering
    # --------------------------------------#
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.eval("../../data/qa/train-eval.csv")
    print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
    print(result)  # EvalResult(eval_loss=0.11738169193267822)
    print(result.loss)  # 0.1173816919326782

```

### test()
Input:
1. input_filepath (string): a path file to a csv file as described in table 3.2


Output: A list of named tuples with keys: "answer", "score", "start" and "end"

The list is in order by ascending csv index. 

#### Table 3.2

1. context (string): background information for answer the question
2. question (string): the question that will be asked 

| context                   | question          | 
|---------------------------|-------------------|
| October 31st is the date  | what is the date? |
| The date is November 23rd | what is the date? | 

#### Example 3.5:
```python
    from happytransformer import HappyQuestionAnswering
    # --------------------------------------#
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.test("../../data/qa/test.csv")
    print(type(result))
    print(result)  # [QuestionAnsweringResult(answer='October 31st', score=0.9939756989479065, start=0, end=12), QuestionAnsweringResult(answer='November 23rd', score=0.967872679233551, start=12, end=25)]
    print(result[0])  # QuestionAnsweringResult(answer='October 31st', score=0.9939756989479065, start=0, end=12)
    print(result[0].answer)  # October 31st

```

#### Example 3.6:
```python
    from happytransformer import HappyQuestionAnswering
    # --------------------------------------#
    happy_qa = HappyQuestionAnswering()
    before_loss = happy_qa.eval("../../data/qa/train-eval.csv").loss
    happy_qa.train("../../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../../data/qa/train-eval.csv").loss
    print("Before loss: ", before_loss)  # 0.11738169193267822
    print("After loss: ", after_loss)  # 0.00037909045931883156
    # Since after_loss < before_loss, the model learned!
    # Note: typically you evaluate with a separate dataset
    # but for simplicity we used the same one 

```