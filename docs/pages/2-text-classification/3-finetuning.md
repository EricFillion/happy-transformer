---
title: Finetuning
parent: Text Classification
nav_order: 3
layout: page
permalink: /text-classification/finetuning/
---

## Text Classification Finetuning 

HappyTextClassification contains three methods for training 
- train(): fine-tune the model to become better at a certain task
- eval(): determine how well the model performs on a labeled dataset
- test(): run the model on an unlabeled dataset to produce predictions  

### train()

inputs: 
1. input_filepath (string): a path file to a csv file as described in table 2.0
2. args (TCTrainArgs): a dataclass. It has the possible values show in table 2.1

#### Table 2.0

1. text (string): text to be classified 
2. label (int): the corresponding label. Must be greater than or equal to 0

| text                          | label |
|-------------------------------|-------|
| Wow what a great place to eat | 1     |
| Horrible food                 | 0     |
| Terrible service              | 0     |
| I'm coming here again         | 1     |


#### Table 2.1
Information about the learning parameters can be found [here](/learning-parameters/)
Information about saving/loading preprocessed data can be found [here](/save-load-data/)

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

#### Example 2.2:
```python
from happytransformer import HappyTextClassification, TCTrainArgs
# --------------------------------------#
happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                   model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                   num_labels=2)  # Don't forget to set num_labels! 
args = TCTrainArgs(num_train_epochs=1)
happy_tc.train("../../data/tc/train-eval.csv", args=args)

```

### eval()
Input:
1. input_filepath (string): a path file to a csv file as described in table 2.1

output:

An object with the field "loss"

#### Example 2.3:
```python
from happytransformer import HappyTextClassification, TCEvalArgs
# --------------------------------------#
happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                   model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                   num_labels=2)  # Don't forget to set num_labels!
args = TCEvalArgs(save_preprocessed_data=False) # for demonstration -- not needed 
result = happy_tc.eval("../../data/tc/train-eval.csv", args=args)
print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
print(result)  # EvalResult(eval_loss=0.007262040860950947)
print(result.loss)  # 0.007262040860950947

```

### test()
Input:
1. input_filepath (string): a path file to a csv file as described in table 2.2

Output: A list of named tuples with keys: "label" and "score"

The list is in order by ascending csv index. 

#### Table 2.2

1. text (string): text that will be classified  

| Text                          |
|-------------------------------|
| Wow what a great place to eat |
| Horrible food                 |
| Terrible service              |
| I'm coming here again         |


#### Example 2.4:
```python
from happytransformer import HappyTextClassification
# --------------------------------------#
happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                   model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                   num_labels=2)  # Don't forget to set num_labels!
result = happy_tc.test("../../data/tc/test.csv")
print(type(result))  # <class 'list'>
print(result)  # [TextClassificationResult(label='POSITIVE', score=0.9998401999473572), TextClassificationResult(label='LABEL_0', score=0.9772131443023682)...
print(type(result[0]))  # <class 'happytransformer.happy_text_classification.TextClassificationResult'>
print(result[0])  # TextClassificationResult(label='POSITIVE', score=0.9998401999473572)
print(result[0].label)  # POSITIVE


```


#### Example 2.5:
```python
from happytransformer import HappyTextClassification
# --------------------------------------#
happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                   model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                   num_labels=2)  # Don't forget to set num_labels!
before_loss = happy_tc.eval("../../data/tc/train-eval.csv").loss
happy_tc.train("../../data/tc/train-eval.csv")
after_loss = happy_tc.eval("../../data/tc/train-eval.csv").loss
print("Before loss: ", before_loss)  # 0.007262040860950947
print("After loss: ", after_loss)  # 0.000162081079906784
# Since after_loss < before_loss, the model learned!
# Note: typically you evaluate with a separate dataset
# but for simplicity we used the same one

```
