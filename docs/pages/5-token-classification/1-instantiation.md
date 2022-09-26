---
title: Token Classification
nav_order: 10
layout: page
permalink: /token-classification/
has_children: true
---
### Token Classification  

Initialize a HappyTokenClassification() object for token classification 

**Initialization Arguments:**
 1. model_type (string): specify the model name in all caps, such as "ROBERTA" or "ALBERT"
 2. model_name(string): potential models can be found [here](https://huggingface.co/models?pipeline_tag=token-classification) 
 3. use_auth_token (string): Specify the authentication token to 
       [load private](https://huggingface.co/transformers/model_sharing.html) models. 
 4. from_tf (bool): Set to True if you want to convert a TensorFlow model to PyTorch model.

#### Example 5.0:
```python
from happytransformer import HappyTokenClassification
# --------------------------------------#
happy_toc = HappyTokenClassification("BERT", "dslim/bert-base-NER")  # default 
happy_toc_large = HappyTokenClassification("XLM-ROBERTA", "xlm-roberta-large-finetuned-conll03-english") 
happy_toc_private = HappyTokenClassification("BERT", "user-repo/bert-base-NER", use_auth_token="123abc")
```
