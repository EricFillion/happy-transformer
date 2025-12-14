
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
from happytransformer import HappyGeneration
# ---------------------------------------------------------
happy_gen = HappyGeneration(model_type="GPT-NEO", model_name="EleutherAI/gpt-neo-125M")
happy_gen.save("model/")
```
### Loading a model

When initializing a Happy Transformer object, provide a path for the model_name parameter. 
#### Example 8.1

```python
from happytransformer import HappyGeneration
# ---------------------------------------------------------
happy_gen = HappyGeneration(model_type="GPT-NEO", model_name="model/")

```
