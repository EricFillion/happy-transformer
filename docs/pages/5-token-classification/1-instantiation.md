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
 

#### Example 5.0:
```python
    from happytransformer import HappyTokenClassification
    # --------------------------------------#
    happy_toc = HappyTokenClassification("BERT", "dslim/bert-base-NER")  # default 
    happy_toc_large = HappyTokenClassification("XLM-ROBERTA", "xlm-roberta-large-finetuned-conll03-english") 
```