# Happy Transformer 

![HappyTransformer](img/HappyTransformer.png)

Happy Transformer is an API built on top of [PyTorch's transformer library](https://pytorch.org/hub/huggingface_pytorch-transformers/)'s that makes it easy to utilize state-of-the-art NLP models. 

## Key Features

  - Available language models: XLNET, BERT and ROBERTA.
  - Predict a masked word within a sentence.
  - Fine tune binary sequence classification models to solve problems like sentiment analysis.
  - Predict the likelihood that sentence B follows sentence A within a paragraph. 
  
  
| Public Methods              | HappyROBERTA | HappyXLNET | HappyBERT |
|-----------------------------|--------------|------------|-----------|
| predict_mask                | ✔            | ✔          | ✔         |
| Sequence Classifier Methods | ✔            | ✔          | ✔         |
| Next Sentence Prediction    |              |            | ✔         |
  
## Installation

```sh
pip install happytransformer
```
## Initialization 

By default base models are used. They are smaller, faster and require significantly less training time
to obtain decent results.

Large models are recommended for tasks that do not require fine tuning such as some word prediction tasks. 

Base models are recommended for tasks that require fine tuning with limited available training data. 

Uncased models do not differentiate between cased and uncased words. For example, the words
"empire" and "Empire" would be reduced to the same token. In comparison, cased models do differentiate between cased and uncased words. 

#### HappyXLNET:

```sh
from happytransformer import HappyXLNET
#--------------------------------------#
xl_base_uncased = HappyXLNET("xlnet-base-uncased")
xl_base_cased = HappyXLNET("xlnet-base-cased")
xl_large_uncased = HappyXLNET("xlnet-large-uncased")
xl_large_cased = HappyXLNET("xlnet-large-cased")
```
#### HappyROBERTA:
```sh
from happytransformer import HappyROBERTA
#--------------------------------------#
happy_roberta_base = HappyROBERTA("roberta-base")
happy_roberta_large = HappyROBERTA("roberta-large")

```
#### HappyBERT :
```sh
from happytransformer import HappyBERT
#--------------------------------------#
bert_base_uncased = HappyBERT("bert-base-uncased")
bert_base_cased = HappyBERT("bert-base-cased")
bert_large_uncased = HappyBERT("bert-large-uncased")
bert_large_cased = HappyBERT("bert-large-cased")
```
## Word Prediction 

Each Happy Transformer has a public  method called "predict_mask(text, options, num_results)" with the following input arguments.
1. Text: the text you wish to predict including a single masked token.
2. options (default = every word): A limited set of words the model can return.
3. num_results (default = 1): The number of returned predictions.

For all Happy Transformers, the masked token is **"[MASK]"**

"predict_mask(text, options, num_results)" returns a list of dictionaries which is exemplified in Example 1 .

It is recommended that you use HappyROBERTA("roberta-large") for masked word prediction.
Avoid using HappyBERT for masked word prediction. 
If you do decide to use HappyXLNET or HappyBERT, then also use their corresponding "large cased model'. 


#### Example 1 :
```sh
from happytransformer import HappyROBERTA
#--------------------------------------#
happy_roberta = HappyROBERTA("roberta-large")
text = "I think therefore I [MASK]"
results = happy_roberta.predict_mask(text)

print(type(results)) # prints: <class 'list'>
print(results) # prints: [{'word': 'am', 'softmax': 0.24738965928554535}]

print(type(results[0])) # prints: <class 'dict'>
print(results[0]) # prints: {'word': 'am', 'softmax': 0.24738965928554535}


```

#### Example 2 :
```sh
from happytransformer import HappyROBERTA
#--------------------------------------#
happy_roberta = HappROBERTA("roberta-large")
text = "To solve world poverty we must invest in [MASK]"
results = happy_roberta.predict_mask(text, num_results = 2)

print(type(results)) # prints: <class 'list'>
print(results) # prints: [{'word': 'education', 'softmax': 0.34365904331207275}, {'word': 'children', 'softmax': 0.03996562585234642}]

print(type(results[0])) # prints: <class 'dict'>
print(results[0]) # prints: {'word': 'education', 'softmax': 0.34365904331207275}


```

#### Example 3 :
```sh
from happytransformer import HappyXLNET
#--------------------------------------#
happy_xlnet = HappyXLNET("xlnet-large-cased")
text = "Can you please pass the [MASK] "
options = ["pizza", "rice", "tofu", 'eggs', 'milk']
results = happy_xlnet.predict_mask(text, options=options, num_results=3)

print(type(results)) # prints: <class 'list'>
print(results) # prints: [{'word': 'tofu', 'softmax': 0.007073382}, {'word': 'pizza', 'softmax': 0.00017212195}, {'word': 'rice', 'softmax': 2.843065e-07}]


print(type(results[1])) # prints: <class 'dict'>
print(results[1]) # prints: {'word': 'pizza', 'softmax': 0.00017212195}

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

### Coming soon
 - Fine tuning for masked word prediction models
 - Question answering models 

License
----
Apache 2
