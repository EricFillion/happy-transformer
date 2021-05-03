---
layout: page
title: Saving and Loading Models
permalink: /save-load-model/
nav_order: 14
---
## Saving and Loading Preprocessed Model

All Happy Transformer objects, such as HappyTextClassification() and HappyGeneration() 
contain functionality that allow models to be saved and loaded. 
  
### save():
The method used to save models. It contains a single argument. 

Inputs: 
1. path: a file path to a directory to save various files. 
    Any previous files of the same names as created files will be overwritten. 
    We recommend that you use an empty directory.  
    

#### Example 8.0 
```python
from happytransformer import HappyTextClassification
# ---------------------------------------------------------
    happy_tc = HappyTextClassification()
    happy_tc.save("model/")
```
### Loading a model

When initializing a Happy Transformer object, provide a path for the load_path parameter. 

#### Example 8.1

```python
from happytransformer import HappyTextClassification
# ---------------------------------------------------------
    happy_tc = HappyTextClassification(load_path="model/")
```
