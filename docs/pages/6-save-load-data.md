---
layout: page
title: Saving and Loading Data
permalink: /save-load-data/
nav_order: 13
---
## Saving and Loading Preprocessed Data

#### Table 7.0

| Parameter                     |Available| 
|-------------------------------|---------|
| Text Generation               |  ✔      |
| Text Classification           |  ✔      | 
| Question Answering            |         |
| Word Prediction               |  ✔      |

The provided data is preprocessed before being given to the model for training or evaluating. 
This process may be computationally expensive depending on the amount of data you're using. 
With Happy Transformer, it is possible to save the the preprocessed data from training or evaluating, 
so that the next time you run the model, you can load the saved data to
skip the preprocessing step.  

All dataclasses for training and evaluating have the following parameters: 

#### Table 7.1

| Parameter                   | Default | Meaning                                |
|-----------------------------|---------|----------------------------------------|
| save_preprocessed_data      | False   | If the preprocessed data will be saved |
| save_preprocessed_data_path | ""      | Path to a json file to save the data   |
| load_preprocessed_data      | False   | If the saved data will be loaded       |
| load_preprocessed_data_path | ""      | Path to a json file to load the data   |

Example 7.0 shows the process of saving and loading a dataset for word prediction training

#### Example 7.0 

```python
from happytransformer import HappyWordPrediction, WPTrainArgs
# ---------------------------------------------------------
happy_wp = HappyWordPrediction()
train_args_1 = WPTrainArgs(save_preprocessed_data=True, save_preprocessed_data_path="data/preprocessed-data.json")
happy_wp.train("data/wp/train-eval.txt", args=train_args_1)
    
    
train_args_2 = WPTrainArgs(load_preprocessed_data=True, load_preprocessed_data_path="data/preprocessed-data.json")
# if you're loading data, then you can set input_filepath to anything 
happy_wp.train(input_filepath="", args=train_args_2)

```

The same pattern is used for all other training and evaluating methods. 
Example 7.1 shows the same process being applied to saving and loading data for 
text classification training. 

#### Example 7.1 

```python
from happytransformer import HappyTextClassification, TCEvalArgs
# ---------------------------------------------------------
happy_tc = HappyTextClassification()
eval_args_1 = TCEvalArgs(save_preprocessed_data=True, save_preprocessed_data_path="data/preprocessed-data.json")
result_1 = happy_tc.eval("data/tc/train-eval.txt", args=eval_args_1)

eval_args_2 = TCEvalArgs(load_preprocessed_data=True, load_preprocessed_data_path="data/preprocessed-data.json")
# if you're loading data, then you can set input_filepath to anything 
result_2 = happy_tc.eval(input_filepath="", args=eval_args_2)

```

Warning: Any file that is located in the path as as save_preprocessed_data_path or load_preprocessed_data_path will be overwritten

 