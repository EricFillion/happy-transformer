[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Downloads](https://pepy.tech/badge/happytransformer)](https://pepy.tech/project/happytransformer)

# Happy Transformer 

Happy Transformer is an package built on top of [Hugging Face's transformer library](https://huggingface.co/transformers/) that makes it easy to utilize state-of-the-art NLP models. 

## Table of Contents
* [News](#News)
* [Features](#Features)
* [Installation](#Installation)
* [Word Prediction](#Word-Prediction)
* [Text Classification](#Text-Classification)
* [Question Answering](#Question-Answering)
* [Question Answering Training](#Question-Answering-Training)
* [Next Sentence Prediction](#Next-Sentence-Prediction)
* [Tech](#Tech)
* [Call For Contributors](#Call-For-Contributors)
* [Maintainers](#Maintainers)

## News: 

### January 12, 2021
**Introducing Version 2.0.0!**

We fully redesigned Happy Transformer from the ground up. 

New Features: 
- Question answering training 
- Multi label text classification training
- Single predictions for text classification 

Deprecated Features: 
- Masked word prediction training
- Masked word prediction with multiple masks 

Breaking changes: 
- Everything

Happy Transformer have been redesigned to promote scalability. 
Now it's easier than ever to add new models and features, and we encourage you
to create PRs to contribute to the project. 


### November 23rd, 2020

Last month, Happy Transformer was presented at a conference called C-Search, and the presentation won the Best Presentation Award.
C-Search is the Queen's University Student Research Conference and had Turing Award Winner Professor Bengio as the Keynote Speaker this year.
The video for the presentation can be found [here](https://www.youtube.com/watch?v=nNdFkq-y8Ng&t=12s). 


### June 9th, 2020
We're happy to announce that we won a Best Paper Award at the Canadian Undergraduate Conference for AI. 
We also received the highest score overall. The paper can be found [here](https://qmind.ca/wp-content/uploads/2020/05/Proceedings-of-CUCAI-2020.pdf) on page 67. 



## Features 
  
| Public Methods                     | Basic Usage  | Training   |
|------------------------------------|--------------|------------|
| Word Prediction                    | ✔            |            |
| Text Classification                | ✔            | ✔          | 
| Question Answering                 | ✔            | ✔          | 
| Next Sentence Prediction           | ✔            |            | 

| Public Methods                     | ALBERT       | BERT       |DISTILBERT |ROBERTA   |
|------------------------------------|--------------|------------|-----------|----------|
| Word Prediction                    | ✔            | ✔          | ✔         |✔         |
| Text Classification                | ✔            | ✔          | ✔         |✔         |
| Question Answering                 | ✔            | ✔          | ✔         |✔         |
| Next Sentence Prediction           | ✔            |            |           |          |
  
## Installation

```sh
pip install happytransformer
```



## Word Prediction
### Initialization  

Initialize a HappyWordPrediction object to perform word prediction. 

**Initialization Arguments:**
 1. model_type (string): either "ALBERT", "BERT" or "DISTILBERT." The default is "DISTILBERT"
 2. model_name(string): below is a URL that contains potential models. 
       [MODELS](https://huggingface.co/models?filter=masked-lm)
 

Note: For all Transformers, the masked token is **"[MASK]"**


We recommend using "HappyWordPrediction("ALBERT", "albert-xxlarge-v2")" for the best performance 


#### Example 1.0:
```python
    from happytransformer import HappyWordPrediction
    # --------------------------------------#
    happy_wp_distilbert = HappyWordPrediction()  # default
    happy_wp_albert = HappyWordPrediction("ALBERT", "albert-base-v2")
    happy_wp_bert = HappyWordPrediction("BERT", "bert-base-uncased")
    happy_wp_roberta = HappyWordPrediction("ROBERTA", "roberta-base")

```


### predict_mask()
The method predict_masks() contains 3 arguments: 
1. text (string): a body of text that contains a single masked token 
2. targets (list of strings): a list of potential answers. All other answers will be ignored 
3. top_k (int): the number of results that will be returned 

Returns: 
A list of objects with fields "token" and "top_k"

Note: if targets are provided, then top_k will be ignored and a score for each target will be returned. 

#### Example 1.1:
```python

from happytransformer import HappyWordPrediction
#--------------------------------------#
    happy_wp = HappyWordPrediction()  # default uses distilbert-base-uncased
    result = happy_wp.predict_mask("I think therefore I [MASK]")
    print(type(result))  # <class 'list'>
    print(result)  # [WordPredictionResult(token='am', score=0.10172799974679947)]
    print(type(result[0]))  # <class 'happytransformer.happy_word_prediction.WordPredictionResult'>
    print(result[0])  # [WordPredictionResult(token='am', score=0.10172799974679947)]
    print(result[0].token)  # am
    print(result[0].score)  # 0.10172799974679947
    

```

#### Example 1.2:
```python

from happytransformer import HappyWordPrediction
#--------------------------------------#
happy_wp = HappyWordPrediction("ALBERT", "albert-xxlarge-v2")
result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", top_k=2)
print(result)  # [WordPredictionResult(token='infrastructure', score=0.09270179271697998), WordPredictionResult(token='healthcare', score=0.07219093292951584)]
print(result[1]) # WordPredictionResult(token='healthcare', score=0.07219093292951584)
print(result[1].token) # healthcare

```

#### Example 1.3:
```python
from happytransformer import HappyWordPrediction
#--------------------------------------#
happy_wp = HappyWordPrediction("ALBERT", "albert-xxlarge-v2")
targets = ["technology", "healthcare"]
result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", targets=targets)
print(result)  # [WordPredictionResult(token='healthcare', score=0.07219093292951584), WordPredictionResult(token='technology', score=0.032044216990470886)]
print(result[1])  # WordPredictionResult(token='technology', score=0.032044216990470886)
print(result[1].token)  # technology


```
## Text Classification 

### Initialization  

Initialize a HappyTextClassification object to perform text classification. 

This model assigns a label to a given text string. For example, you can train a model to 
detect if an email is spam based on its text. 


**Initialization Arguments:** 
1. model_type (string): either "ALBERT", "BERT" or "DISTILBERT." The default is "DISTILBERT"
2. model_name(string): below is a URL that contains potential models. The default is "distilbert-base-uncased"
       [MODELS](https://huggingface.co/models?filter=text-classification)
3. num_labels(int): The number of text categories. The default is 2
    
WARNING: If you try to load a pretrained model that has a different number of categories 
than num_labels, then you will get an error 

NOTE: "albert-base-v2", "bert-base-uncased" and "distilbert-base-uncased" do not have a predefined 
number of labels, so if you use these models you can set num_labels freely 


#### Example 2.0:
```python
    from happytransformer import HappyTextClassification
    # --------------------------------------#
    happy_tc_distilbert = HappyTextClassification()  # default with "distilbert-base-uncased" and num_labels=2
    happy_tc_albert = HappyTextClassification(model_type="ALBERT", model_name="albert-base-v2")
    happy_tc_bert = HappyTextClassification("BERT", "bert-base-uncased")
    happy_tc_roberta = HappyTextClassification("ROBERTA", "roberta-base")

```


### classify_text()

Input: 
1. text (string): Text that will be classified 

Returns: 
An object with fields "label" and "score"

#### Example 2.1:
```python
    from happytransformer import HappyTextClassification
    # --------------------------------------#
    happy_tc = HappyTextClassification(model_type="DISTILBERT",  model_name="distilbert-base-uncased-finetuned-sst-2-english")
    result = happy_tc.classify_text("Great movie! 5/5")
    print(type(result))  # <class 'happytransformer.happy_text_classification.TextClassificationResult'>
    print(result)  # TextClassificationResult(label='LABEL_1', score=0.9998761415481567)
    print(result.label)  # LABEL_1

```



## Text Classification Training

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
## Question Answering

### Initialization  
Initialize a HappyQuestionAnswering object to perform question answering. 

This model answers a question given a body of that's text relevant to the questions. 

The outputted answer is always a text-span with the provided information. 

**Initialization Arguments:**
1. model_type (string): either "ALBERT", "BERT" or "DISTILBERT." The default is "DISTILBERT"
2. model_name(string): below is a URL that contains potential models. 
   [MODELS](https://huggingface.co/models?filter=question-answering)


We recommend using "HappyQuestionAnswering("ALBERT", "mfeb/albert-xxlarge-v2-squad2")" for the best performance 


#### Example 3.0:
```python
    from happytransformer import HappyQuestionAnswering
    # --------------------------------------#
    happy_qa_distilbert = HappyQuestionAnswering()  # default
    happy_qa_albert = HappyQuestionAnswering("ALBERT", "mfeb/albert-xxlarge-v2-squad2")
    # good model when using with limited hardware 
    happy_qa_bert = HappyQuestionAnswering("BERT", "mrm8488/bert-tiny-5-finetuned-squadv2")
    happy_qa_roberta = HappyQuestionAnswering("ROBERTA", "deepset/roberta-base-squad2")

```


### answer_question()

Inputs: 
1. context (string): background information, which contains a text-span that is the answer 
2. question (string): the question that will be asked 
3. top_k (int): the number of results that will be returned (default=1)

Returns: 
 A list of a objects with fields: "answer", "score", "start" and "end." 
The list is in descending order by score

#### Example 3.1:
```python
    from happytransformer import HappyQuestionAnswering
    # --------------------------------------#
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.answer_question("Today's date is January 10th, 2021", "What is the date?")
    print(type(result))  # <class 'list'>
    print(result)  # [QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34)]
    print(type(result[0]))  # <class 'happytransformer.happy_question_answering.QuestionAnsweringResult'>
    print(result[0])  # QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34)
    print(result[0].answer)  # January 10th, 2021
```

#### Example 3.2:
```python
    from happytransformer import HappyQuestionAnswering
    # --------------------------------------#
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.answer_question("Today's date is January 10th, 2021", "What is the date?", top_k=2)
    print(type(result))  # <class 'list'>
    print(result)  # [QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34), QuestionAnsweringResult(answer='January 10th', score=0.017306014895439148, start=16, end=28)]
    print(result[1].answer)  # January 10th

```


## Question Answering Training

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
## Next Sentence Prediction 

### Initialization  

Initialize a HappyNextSentence object to next sentence prediction  

**Initialization Arguments:**
 1. model_type (string): The default is "BERT", which is currently the only available model 
 2. model_name(string): We recommend  none-finetuned BERT models like 
 "bert-base-uncased" and "bert-large-uncased"
 

#### Example 4.0:
```python
    from happytransformer import HappyNextSentence
    # --------------------------------------#
    happy_ns = HappyNextSentence()  # default is "bert-base-uncased"
    happy_ns_large = HappyNextSentence("BERT", "bert-large-uncased") 

```

### predict_next_sentence()

Inputs: 
We recommend keeping sentence_a and sentence_b to a single sentence. But longer inputs still work. 
1. sentence_a (string): A sentence 
2. sentence_b (string): A sentence that may or may not follow sentence_a

Returns: 
A float between 0 and 1 that represents how likely sentence_a follows sentence_b. 

#### Example 4.1:
```python
    from happytransformer import HappyNextSentence
    # --------------------------------------#
    happy_ns = HappyNextSentence()
    result = happy_ns.predict_next_sentence(
        "How old are you?",
        "I am 21 years old."
    )
    print(type(result))  # <class 'float'>
    print(result)  # 0.9999918937683105
```

#### Example 4.2:
```python
    from happytransformer import HappyNextSentence
    # --------------------------------------#
    happy_ns = HappyNextSentence()
    result = happy_ns.predict_next_sentence(
        "How old are you?",
        "Queen's University is in Kingston Ontario Canada"
    )
    print(type(result))  # <class 'float'>
    print(result)  # 0.00018497584096621722
```
## Tech

 Happy Transformer uses a number of open source projects:

* [transformers](https://github.com/huggingface/transformers/stargazers) - State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch!
* [pytorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python
* [tqdm](https://github.com/tqdm/tqdm) - A Fast, Extensible Progress Bar for Python and CLI

 HappyTransformer is also an open source project with this [public repository](https://github.com/EricFillion/happy-transformer)
 on GitHub. 
 
### Call for contributors 
 Happy Transformer is a growing API. We're seeking more contributors to help accomplish our mission of making state-of-the-art AI easier to use.  

### Maintainers
- [Eric Fillion](https://github.com/ericfillion)  Lead Maintainer
- [Ted Brownlow](https://github.com/ted537) Maintainer
