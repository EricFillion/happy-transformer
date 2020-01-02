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
happy_xlnet = HappyXLNET()
```
###### HappyROBERTA:
```sh
from happytransformer import HappyROBERTA
happy_roberta = HappyROBERTA()
```
###### HappyBERT :
```sh
from happytransformer import HappyBERT
happy_bert = HappyBERT()
```
## Word Prediction 

Each Happy Transformer has a public  method called "predict_mask" with the following arguments 
1. Text: the text you wish to predict including the masked token
2. options (default = every word): A limited set of words the model can return 
3. k (default = 1): The number of returned predictions 
For all Happy Transformers, the masked token is "[MASK]"

###### Example 1:
```sh
from happytransformer import HappyXLNET
#--------------------------------------#
happy_xlnet = HappyXLNET()
text = "To stop global warming we must invest in [MASK] power generation"
result = happy_xlnet.predict_mask(text)
print(type(result)) # returns: 
print(result) # returns: 
```
###### Example 2 :
```sh
from happytransformer import HappyROBERTA
#--------------------------------------#
happy_roberta = HappyROBERTA()
text = "To stop global warming we must invest in [MASK] power generation"
options = ["wind", "oil", "solar", 'nuclear', 'wind']
result = happy_roberta.predict_mask(text, options)
print(type(result) # returns :
print(result) # returns: 
```
###### Example 3 :
```sh
from happytransformer import HappyBERT
#--------------------------------------#
happy_bert = HappyBERT()
text = "To stop global warming we must invest in [MASK] power generation"
options = ["wind", "oil", "solar", 'nuclear', 'wind']
result = happy_bert.predict_mask(text, options, 3)
print(type(result) # returns :
print(result) # returns: 
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
print(type(eval_results)) # Output: 
print(eval_results) # Output: 

test_csv_path = "data/test.csv"
test_results = happy_roberta.test_sequence_classifier(test_csv_path)
print(type(test_results)) # Output: 
print(test_results) # Output: 
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

### Tech

 Happy Transformer uses a number of open source projects to work properly:

* [transformers](https://github.com/huggingface/transformers/stargazers) - State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch!
*  [pytorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python
*  [scikit-learn](https://github.com/scikit-learn/scikit-learn) - A set of python modules for machine learning and data mining
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


