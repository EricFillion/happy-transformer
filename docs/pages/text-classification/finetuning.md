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
1. input_filepath (string): a path file to a csv file as described in table 2.1
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
 
#### Table 2.1

1. text (string): text to be classified 
2. label (int): the corresponding label

| Text                          | label |
|-------------------------------|-------|
| Wow what a great place to eat | 1     |
| Horrible food                 | 0     |
| Terrible service              | 0     |
| I'm coming here again         | 1     |

#### Example 2.3:
```python
    from happytransformer import HappyTextClassification
    # --------------------------------------#
     happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                       model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                       num_labels=2)  # Don't forget to set num_labels! 
    happy_tc.train("../../data/tc/train-eval.csv")

```

### eval()
Input:
1. input_filepath (string): a path file to a csv file as described in table 2.1

output:

An object with the field "loss"

#### Example 2.3:
```python
    from happytransformer import HappyTextClassification
    # --------------------------------------#
    happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                       model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                       num_labels=2)  # Don't forget to set num_labels!
    result = happy_tc.eval("../../data/tc/train-eval.csv")
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
    print(result)  # [TextClassificationResult(label='LABEL_1', score=0.9998401999473572), TextClassificationResult(label='LABEL_0', score=0.9772131443023682)...
    print(type(result[0]))  # <class 'happytransformer.happy_text_classification.TextClassificationResult'>
    print(result[0])  # TextClassificationResult(label='LABEL_1', score=0.9998401999473572)
    print(result[0].label)  # LABEL_1


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