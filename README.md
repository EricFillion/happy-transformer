# Happy Transformer 

![HappyTransformer](img/HappyTransformer.png)

Happy Transformer is an API built on top of PyTorch's transformer library that makes it easy to utilize state-of-the-art NLP models. 
## Key Features!

  -  Available language models: XLNet, BERT and RoBERTa 
  -  Predict masked words within sentences 
  - Fine tune binary sequence classification models to solve problems like sentiment analysis 
  - Predict the likelihood that sentence B follows sentence A within a paragraph. 
  
# Installation

```sh
pip install happytransformer
```
## Initialization 
###### HappyXLNET:

```sh
from happytransformer import HappyXLNET
#--------------------------------------#
xl_base_uncased = HappyXLNET("xlnet-base-uncased")
xl_base_cased = HappyXLNET("xlnet-base-cased")
xl_large_uncased = HappyXLNET("xlnet-large-uncased")
xl_large_cased = HappyXLNET("xlnet-large-cased")
```
###### HappyROBERTA:
```sh
from happytransformer import HappyROBERTA
#--------------------------------------#
happy_roberta_base = HappyROBERTA("roberta-base")
happy_roberta_large = HappyROBERTA("roberta-large")

```
###### HappyBERT :
```sh
from happytransformer import HappyBERT
#--------------------------------------#
bert_base_uncased = HappyBERT("bert-base-uncased")
bert_base_cased = HappyBERT("bert-base-cased")
bert_large_uncased = HappyBERT("bert-large-uncased")
bert_large_cased = HappyBERT("bert-large-cased")
```
## Word Prediction 

Each Happy Transformer has a public  method called "predict_mask" with the following arguments 
1. Text: the text you wish to predict including the masked token
2. options (default = every word): A limited set of words the model can return 
3. k (default = 1): The number of returned predictions 
For all Happy Transformers, the masked token is "[MASK]"

 "predict_mask" returns a list of dictionaries which is exemplified below 

It is recommended that you use HappyROBERTA("roberta-large") for masked word prediction.
Avoid using HappyBERT for masked word prediction. 
If you do decided to use HappyXLNET or HappyBERT, then also use their corresponding "large cased model'. 


###### Example 1 :
```sh
from happytransformer import HappyROBERTA
#--------------------------------------#
happy_roberta = HappROBERTA("roberta-large")
text = "I think therefore I [MASK]"
results = happy_roberta.predict_mask(text)

print(type(results)) # prints: <class 'list'>
print(results) # prints: [{'word': 'am', 'softmax': 0.24738965928554535}]

print(type(results[0])) # prints: <class 'dict'>
print(results[0]) # prints: {'word': 'am', 'softmax': 0.24738965928554535}


```



###### Example 2 :
```sh
from happytransformer import HappyROBERTA
#--------------------------------------#
happy_roberta = HappROBERTA("roberta-large")
text = "To solve world poverty we must invest in [MASK]"
results = happy_roberta.predict_mask(text, k = 2)

print(type(results)) # prints: <class 'list'>
print(results) # prints: [{'word': 'education', 'softmax': 0.34365904331207275}, {'word': 'children', 'softmax': 0.03996562585234642}]

print(type(results[0])) # prints: <class 'dict'>
print(results[0]) # prints: {'word': 'education', 'softmax': 0.34365904331207275}


```


###### Example 3 :
```sh
from happytransformer import HappyXLNET
#--------------------------------------#
happy_xlnet = HappyXLNET("xlnet-large-cased")
text = "Can you please pass the [MASK] "
options = ["pizza", "rice", "tofu", 'eggs', 'milk']
results = happy_xlnet.predict_mask(text, options=options, k=3)

print(type(results)) # prints: <class 'list'>
print(results) # prints: [{'word': 'tofu', 'softmax': 0.007073382}, {'word': 'pizza', 'softmax': 0.00017212195}, {'word': 'rice', 'softmax': 2.843065e-07}]


print(type(results[1]))# prints: <class 'dict'>
print(results[1]) # prints: {'word': 'pizza', 'softmax': 0.00017212195}



```
## Binary Sequence Classification 

Binary sequence classification (BSC) has many applications. For example, by using BSC, you can train a model to predict if a yelp review is positive or negative. Another example includes determining if an email is spam or ham. 

Each Happy Transformer has four methods that are utilized  for binary sequence classification.

They include, init_sequence_classifier(), custom_init_sequence_classifier(args), train_sequence_classifier(train_csv_path), and eval_sequence_classifier(eval_csv_path)

Before we explore each method in depth, here is an example that shows how easily these methods can be used to accomplish binary sequence classification tasks. 
###### Example 1:
```sh
from happytransformer import HappyROBERTA
#------------------------------------------#
happy_roberta = HappyROBERTA()
happy_roberta.init_sequence_classifier()

train_csv_path = "data/train.csv"
happy_roberta.train_sequence_classifier(train_csv_path)

eval_csv_path = "data/eval.csv"
eval_results = happy_roberta.eval_sequence_classifier(eval_csv_path)
print(type(eval_results)) # prints: 
print(eval_results) # prints: 

test_csv_path = "data/test.csv"
test_results = happy_roberta.test_sequence_classifier(test_csv_path)
print(type(test_results)) # prints: 
print(test_results) # prints: 
```
##### init_sequence_classifier()
Initialize binary sequence classification for the Happy Transformer object with the default settings.

##### custom_init_sequence_classifier(args)

Takes in a dictionary with custom settings for inializing and training the transformer. 
Called instead of init_sequence_classifier(). 

###### default classifier arguments
 
```
# found under "from happytransformer.sequence_classification.classifier_args"
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
###### Example 2 :
 ```sh
from happytransformer import HappyROBERTA
from happytransformer.seq_class.classifier_args import classifier_args
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

The HappyBERT Transformer has a publich method called "is_next_sentence" which can be used for next Sentence Prediction tasks.
The method takes the following arguments:
1. A: the first sentence in question
2. B: the second sentence in question

The method takes the two sentences and determines the likelihood that sentence B follows sentence A.
This likelihood is returned as a tuple where the first element is True or False, indicating if it is true that sentence B follows sentence A. The second element of the tuple is the softmax from the Next Sentence Prediction transformer which can be used to determine the confidence of the model in the answer.

###### Example 1 :
```sh
from happytransformer import HappyBERT
#--------------------------------------#
happy_bert = HappyBERT()
sentence_a = "How old are you?"
sentence_b = "I am 93 years old."
sentence_c = "The Eiffel Tower is in Paris."
result = happy_bert.is_next_sentence(sentence_a, sentence_b)
print(type(result)) # prints: <class 'tuple'>
print(result) # prints: (True, 0.9999142289161682)
result = happy_bert.is_next_sentence(sentence_a, sentence_c)
print(type(result)) # prints: <class 'tuple'>
print(result) # prints: (False, 0.9988276362419128) 
```
### Tech

 Happy Transformer uses a number of open source projects to work properly:

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
MIT


