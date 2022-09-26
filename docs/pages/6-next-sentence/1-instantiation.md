---
title: Next Sentence Prediction 
nav_order: 11
layout: page
permalink: /next-sentence-prediction/
has_children: true
---


## Next Sentence Prediction  

Initialize a HappyNextSentence() object to next sentence prediction  

**Initialization Arguments:**
 1. model_type (string): The default is "BERT", which is currently the only available model 
 2. model_name(string): We recommend  BERT models like 
 "bert-base-uncased" and "bert-large-uncased" that have not been finetuned  
 3. use_auth_token (string): Specify the authentication token to 
       [load private](https://huggingface.co/transformers/model_sharing.html) models. 
 4. from_tf (bool): Set to True if you want to convert a TensorFlow model to PyTorch model.

#### Example 6.0:
```python
from happytransformer import HappyNextSentence
# --------------------------------------#
happy_ns = HappyNextSentence("BERT", "bert-base-uncased")  # default 
happy_ns_large = HappyNextSentence("BERT", "bert-large-uncased") 
happy_ns_private = HappyNextSentence("BERT", "user-repo/bert-base-uncased", use_auth_token="123abc")
```
