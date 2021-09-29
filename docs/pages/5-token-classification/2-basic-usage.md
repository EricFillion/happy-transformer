---
title: Usage
parent: Token Classification
nav_order: 1
layout: page
permalink: /token-classification/usage/
---
## Token Classification Basic Usage

### Initialization  

Initialize a HappyTokenClassification object for token classification  

**Initialization Arguments:**
 1. model_type (string): specify the model name in all caps, such as "ROBERTA" or "ALBERT"
 2. model_name(string): potential models can be found [here](https://huggingface.co/models?pipeline_tag=token-classification)
 

### classify_token()

Inputs: 
1. sentence_a (string): Text you wish to classify. Be sure to provide full sentences rather than individual words so that the model has more context.  

Returns: 
A list of objects with the following fields: 
    word: The classified word 
    score: the probability of the entity 
    entity: the predicted entity. Each model has it's own unique set of entities. 
    index: The index of the token within the tokenized text 
    start: The index of the string where the first letter of the predicted word occurs 
    end: The index of the string where the last letter of the predicted word occurs 



#### Example 5.1:
```python
from happytransformer import HappyTokenClassification
# --------------------------------------#
happy_toc = HappyTokenClassification(model_type="BERT", model_name="dslim/bert-base-NER")
result = happy_toc.classify_token("My name is Geoffrey and I live in Toronto")
print(type(result))  # <class 'list'>
print(result[0].word)  # Geoffrey
print(result[0].entity)  # B-PER
print(result[0].score)  # 0.9988969564437866
print(result[0].index)  # 4
print(result[0].start) # 11
print(result[0].end)  # 19
print(result[1].word)  # Toronto
print(result[1].entity)  # B-LOC

```