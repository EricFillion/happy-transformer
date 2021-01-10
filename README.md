[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Downloads](https://pepy.tech/badge/happytransformer)](https://pepy.tech/project/happytransformer)

# Happy Transformer 

## Table of Contents
* [News](#News)
* [Features](#Features)
* [Installation](#Installation)
* [Word Prediction](#Word-Prediction)
* [Text Classification](#Binary-Sequence-Classification)
* [Next Sentence Prediction](#Next-Sentence-Prediction)
* [Question Answering](#Question-Answering)
* [Tech](#Tech)
* [Call For Contributors](#Call-For-Contributors)
* [Maintainers](#Maintainers)

## News: 

### January x, 2021
Introducing Version 2.0.0! 
...

### November 23rd, 2020

Last month, Happy Transformer was presented at a conference called C-Search, and the presentation won the Best Presentation Award. C-Search is the Queen's University Student Research Conference and had Turing Award Winner Professor Bengio as the Keynote Speaker this year. The video for the presentation can be found [here](https://www.youtube.com/watch?v=nNdFkq-y8Ng&t=12s). 



### June 9th, 2020
We're happy to announce that we won a Best Paper Award at the Canadian Undergraduate Conference for AI. We also received the highest score overall. The paper can be found [here](https://qmind.ca/wp-content/uploads/2020/05/Proceedings-of-CUCAI-2020.pdf) on page 67. 


Happy Transformer is an package built on top of [Hugging Face's transformer library](https://huggingface.co/transformers/) that makes it easy to utilize state-of-the-art NLP models. 

## Features 
  
| Public Methods                     | Basic Usage  | Training   |
|------------------------------------|--------------|------------|
| Text Classification                | ✔            | ✔          | 
| Question Answering                 | ✔            | ✔          | 
| Word Prediction                    | ✔            |            |
| Next Sentence Prediction           | ✔            |            | 

| Public Methods                     | ALBERT       | BERT       |DISTILBERT |
|------------------------------------|--------------|------------|-----------|
| Text Classification                | ✔            | ✔          | ✔         |
| Question Answering                 | ✔            | ✔          | ✔         |
| Word Prediction                    | ✔            | ✔          | ✔         |
| Next Sentence Prediction           | ✔            | ✔          | ✔         |
  
## Installation

```sh
pip install happytransformer
```



## Word Prediction

Initialize a HappyWordPrediction() object to perform word prediction. 

Initialization Arguments: 
    1. model_type (string): either "ALBERT", "BERT" or "DISTILBERT." The default is "DISTILBERT"
    2. model_name(string): below is a URL that contains potential models. 
       [MODELS](https://huggingface.co/models?filter=masked-lm)
 

For all Transformers, the masked token is **"[MASK]"**

### Initialization  

We recommend using "HappyWordPrediction("ALBERT", "albert-xxlarge-v2")" for the best performance 


#### Example 1.0:
```python
    from happytransformer import HappyWordPrediction
    # --------------------------------------#
    happy_wp_distilbert = HappyWordPrediction()  # default
    happy_wp_albert = HappyWordPrediction("ALBERT", "albert-base-v2")
    happy_wp_bert = HappyWordPrediction("BERT", "bert-base-uncased")

```


### predict_mask()
The method predict_masks() contains 3 arguments: 
1. text (string): a body of text that contains a single masked token 
2. targets (list of strings): a list of potential answers. All other answers will be ignored 
3. top_k (int): the number of results that will be returned 

Returns: 
A list of named tuples with arguments: "token_str" and "top_k"

Note: if targets are provided, then top_k will be ignored and a score for each target will be returned. 

#### Example 1.1:
```python

from happytransformer import HappyWordPrediction
#--------------------------------------#
    happy_wp = HappyWordPrediction()  # default uses distilbert-base-uncased
    result = happy_wp.predict_mask("I think therefore I [MASK]")
    print(type(result))  # <class 'list'>
    print(result)  # [WordPredictionResult(token_str='am', score=0.10172799974679947)]
    print(type(result[0]))  # <class 'list'>
    print(result[0])  # [WordPredictionResult(token_str='am', score=0.10172799974679947)]
    print(result[0].token_str)  # am
    print(result[0].score)  # 0.10172799974679947
    

```

#### Example 1.2:
```python

from happytransformer import HappyWordPrediction
#--------------------------------------#
happy_wp = HappyWordPrediction("ALBERT", "albert-xxlarge-v2")
result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", top_k=2)
print(result)  # [WordPredictionResult(token_str='infrastructure', score=0.09270179271697998), WordPredictionResult(token_str='healthcare', score=0.07219093292951584)]
print(result[1]) # WordPredictionResult(token_str='healthcare', score=0.07219093292951584)
print(result[1].token_str) # healthcare

```

#### Example 1.3:
```python
from happytransformer import HappyWordPrediction
#--------------------------------------#
happy_wp = HappyWordPrediction("ALBERT", "albert-xxlarge-v2")
targets = ["technology", "healthcare"]
result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", targets=targets)
print(result)  # [WordPredictionResult(token_str='healthcare', score=0.07219093292951584), WordPredictionResult(token_str='technology', score=0.032044216990470886)]
print(result[1])  # WordPredictionResult(token_str='technology', score=0.032044216990470886)
print(result[1].token_str)  # technology


```

## Binary Sequence Classification 

Binary sequence classification (BSC) has many applications. For example, by using BSC, you can train a model to predict if a yelp review is positive or negative. 
Another example includes determining if an email is spam or ham. 

Each Happy Transformer has four methods that are utilized for binary sequence classification:

1. init_sequence_classifier()
2. custom_init_sequence_classifier(args)
3. train_sequence_classifier(train_csv_path)
4. eval_sequence_classifier(eval_csv_path)


### init_sequence_classifier()
Initialize binary sequence classification for the HappyTransformer object with the default settings.


### train_sequence_classifier(train_csv_path):
Trains the HappyTransformer's sequence classifier.

One of the two init sequence classifier methods must be called before this method can be called.

Argument:

    1. train_csv_path: A string directory path to the csv that contains the training data.

##### train_csv requirements: 
    1. The csv must contain *NO* header. 
    2. Each row contains a training case. 
    3. The first column contains either a 0 or a 1 to indicate whether the training case is for case "0" or case "1". 
    4. The second column contains the text for the training case
#### Example 1
|   |                                                              | 
|---|--------------------------------------------------------------| 
| 0 |  Terrible service and awful food                             | 
| 1 |  My new favourite Chinese restaurant!!!!                     | 
| 1 |  Amazing food and okay service. Overall a great place to eat | 
| 0 |  The restaurant smells horrible.                             | 

This method does not return anything 


### eval_sequence_classifier(eval_csv_path):
Evaluates the trained model against an input.

train_sequence_classifier(train_csv_path): must be called before this method can be called.

Argument:

    1. eval_csv_path: A string directory path to the csv that contains the evaluating data.

##### eval_csv requirements: (same as train_csv requirements) 
    1. The csv must contain *NO* header. 
    2. Each row contains a training case. 
    3. The first column contains either a 0 or a 1 to indicate whether the training case is for case "0" or case "1". 
    4. The second column contains the text for the training case
    
**Returns** a python dictionary that contains a count for the following values

*true_positive:* The model correctly predicted the value 1 .
*true_negative:* The model correctly predicted the value 0.
*false_positive':* The model incorrectly predicted the value 1.
*false_negative* The model incorrectly predicted the value 0.


### test_sequence_classifier(test_csv_path):
Tests the trained model against an input.

train_sequence_classifier(train_csv_path): must be called before this method can be called.

Argument:
    1. test_csv_path: A string directory path to the csv that contains the testing data

##### test_csv requirements: 
    1. The csv must contain *NO* header. 
    2. Each row contains a single test case. 
    3. The csv contains a single column with the text for each test case.

#### Example 2:

|                                           | 
|-------------------------------------------| 
| 5 stars!!!                                | 
| Cheap food at an expensive price          | 
| Great location and nice view of the ocean | 
| two thumbs down                           | 

**Returns** a list of integer values in ascending order by test case row index.
For example, for the csv file shown in Example 2, the result would be [1, 0, 1, 0]. 
Where the first index in the list  corresponds to "5 stars!!!" 
and the last index corresponds to "two thumbs down."


#### Example 3:
```sh
from happytransformer import HappyROBERTA
#------------------------------------------#
happy_roberta = HappyROBERTA()
happy_roberta.init_sequence_classifier()

train_csv_path = "data/train.csv"
happy_roberta.train_sequence_classifier(train_csv_path)

eval_csv_path = "data/eval.csv"
eval_results = happy_roberta.eval_sequence_classifier(eval_csv_path)
print(type(eval_results)) # prints: <class 'dict'>
print(eval_results) # prints: {'true_positive': 300', 'true_negative': 250, 'false_positive': 40, 'false_negative': 55}

test_csv_path = "data/test.csv"
test_results = happy_roberta.test_sequence_classifier(test_csv_path)
print(type(test_results)) # prints: <class 'list'>
print(test_results) # prints: [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0 ]
```


### custom_init_sequence_classifier(args)

Initializing the sequence classifier with custom settings. 
Called instead of init_sequence_classifier(). 
argument:
    1. args: a python dictionary that contains all of the same fields as the default arguments

### default classifier arguments
 
```
# found under "from happytransformer.classifier_args"
classifier_args = {
    # Basic fine tuning parameters
    'learning_rate': 1e-5,
    'num_epochs': 2,
    'batch_size': 8,

    # More advanced fine tuning parameters
    'max_seq_length': 128,  #  Max number of tokens per input. Max value = 512
    'adam_epsilon': 1e-5,
    'gradient_accumulation_steps': 1,
    'weight_decay': 0,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    # More modes will become available in future releases
    'task_mode': 'binary',
    }
 ```
#### Example 4:
 ```sh
from happytransformer import HappyROBERTA
from happytransformer import classifier_args
#------------------------------------------#
happy_xlnet = HappyXLNET()

custom_args = classifier_args.copy()
custom_args["learning_rate"] = 2e-5
custom_args['num_epochs'] = 4
custom_args["batch_size"] = 3

happy_xlnet.custom_init_sequence_classifier(custom_args)
# Continue from example 1 after "happy_roberta.init_sequence_classifier()""

```


## Next Sentence Prediction

*Determine the likelihood that sentence B follows sentence A.*


**HappyBERT** has a method called "predict_next_sentence" which is used for next sentence prediction tasks.
The method takes the following arguments:

    1. sentence_a: A **single** sentence in a body of text
    2. sentence_b: A **single** sentence that may or may not follow sentence sentence_a

This likelihood that sentence_b follows sentenced_a is returned as a boolean value that is either True or False indicating if it is true that sentence B follows sentence A.  

###### Example 1:
```sh
from happytransformer import HappyBERT
#--------------------------------------#
happy_bert = HappyBERT()
sentence_a = "How old are you?"
sentence_b = "I am 93 years old."
sentence_c = "The Eiffel Tower is in Paris."
result = happy_bert.predict_next_sentence(sentence_a, sentence_b)
print(type(result)) # prints: <class 'bool'>
print(result) # prints: True
result = happy_bert.predict_next_sentence(sentence_a, sentence_c)
print(type(result)) # prints: <class 'bool'>
print(result) # prints: False
```
###### Example 2: (New feature)
You can now set the use_probability parameter to True to make the next_sentence function output 
a probability instead of a boolean answer. 
```sh
from happytransformer import HappyBERT
#--------------------------------------#
happy_bert = HappyBERT()
sentence_a = "How old are you?"
sentence_b = "I am 93 years old."
result = happy_bert.predict_next_sentence(sentence_a, sentence_b,  use_probability=True)
print(type(result)) # prints: <class 'float'>
print(result) # prints: 0.999990701675415


```

## Question Answering

*Determine the answer to a given question using a body of supplied text.*

##### Single Answer
**HappyBERT** has a method called "answer_question" which is used for question answering tasks.
The method takes the following arguments:

    1. question: The question to be answered
    2. text: The text containing the answer to the question

The output from the method is the answer to the question, returned as a string.

###### Example 1:
```sh
from happytransformer import HappyBERT
#--------------------------------------#
happy_bert = HappyBERT()
question = "Who does Ernie live with?"
text = "Ernie is an orange Muppet character on the long running PBS and HBO children's television show Sesame Street. He and his roommate Bert form the comic duo Bert and Ernie, one of the program's centerpieces, with Ernie acting the role of the naïve troublemaker and Bert the world weary foil."  # Source: https://en.wikipedia.org/wiki/Ernie_(Sesame_Street)
result = happy_bert.answer_question(question, text)
print(type(result)) # prints: <class 'str'>
print(result) # prints: bert
```

##### Multiple Answers
**HappyBERT** has a method called "answers_to_question" which is used to generate multiple answers for a single question

    1. question: The question to be answered
    2. text: The text containing the answer to the question
    3. k: The number of answers that will be returned 

The output is a list of  dictionaries. 
Each dictionary contains two keys: text and softmax. 
The text key contains the answer in the form of a string. 
The softmax key contains the "probability" of the answer as a float between 0 and 1. 

###### Example 1:
```sh
from happytransformer import HappyBERT
#--------------------------------------#
happy_bert = HappyBERT()
question = "Who does Ernie live with?"
text = "Ernie is an orange Muppet character on the long running PBS and HBO children's television show Sesame Street. He and his roommate Bert form the comic duo Bert and Ernie, one of the program's centerpieces, with Ernie acting the role of the naïve troublemaker and Bert the world weary foil."  # Source: https://en.wikipedia.org/wiki/Ernie_(Sesame_Street)
result = happy_bert.answers_to_question(question, text, k=3)
print(type(result)) # prints: <class 'list'>
print(result) # prints: [{'text': 'bert', 'softmax': 0.9916905164718628}, {'text': 'roommate bert', 'softmax': 0.004403269849717617}, {'text': 'his roommate bert', 'softmax': 0.0039062034338712692}]


best_answer = result[0]
second_best_answer = result[1]

print(type(best_answer)) # prints:<class 'dict'>

print(best_answer) # prints: {'text': 'bert', 'softmax': 0.9916905164718628}

print(best_answer["text"]) # prints: bert
print(best_answer["softmax"]) # prints: 0.9916905164718628

```


## Tech

 Happy Transformer uses a number of open source projects:

* [transformers](https://github.com/huggingface/transformers/stargazers) - State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch!
*  [pytorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python
* [scikit-learn](https://github.com/scikit-learn/scikit-learn) - A set of python modules for machine learning and data mining
* [numpy](https://github.com/numpy/numpy) - Array computation 
* [pandas](https://github.com/pandas-dev/pandas) - Powerful data structures for data analysis, time series, and statistics
* [tqdm](https://github.com/tqdm/tqdm) - A Fast, Extensible Progress Bar for Python and CLI
*  [pytorch-transformers-classification](https://github.com/ThilinaRajapakse/pytorch-transformers-classification) - Text classification for BERT, RoBERTa, XLNet and XLM

 HappyTransformer is also an open source project with this [public repository](https://github.com/EricFillion/happy-transformer)
 on GitHub. 
 
### Call for contributors 
 Happy Transformer is a new and growing API. We're seeking more contributors to help accomplish our mission of making state-of-the-art AI easier to use.  

### Maintainers
- [Eric Fillion](https://github.com/ericfillion)  Lead Maintainer
- [Ted Brownlow](https://github.com/ted537) Maintainer
