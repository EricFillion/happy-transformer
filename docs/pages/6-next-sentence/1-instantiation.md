---
title: Next Sentence Prediction 
nav_order: 11
layout: page
permalink: /next-sentence-prediction/
has_children: true
---


## Next Sentence Prediction  

Initialize a HappyNextSentence object to next sentence prediction  

**Initialization Arguments:**
 1. model_type (string): The default is "BERT", which is currently the only available model 
 2. model_name(string): We recommend  none-finetuned BERT models like 
 "bert-base-uncased" and "bert-large-uncased"
 

#### Example 4.0:
```python
    from happytransformer import HappyNextSentence
    # --------------------------------------#
    happy_ns = HappyNextSentence("BERT", "bert-base-uncased")  # default 
    happy_ns_large = HappyNextSentence("BERT", "bert-large-uncased") 

```