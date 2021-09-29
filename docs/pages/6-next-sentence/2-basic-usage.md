---
title: Usage
parent: Next Sentence Prediction
nav_order: 1
layout: page
permalink: /next-sentence-prediction/usage/
---


## Next Sentence Prediction Basic Usage

### predict_next_sentence()

Inputs: 
We recommend keeping sentence_a and sentence_b to a single sentence. But longer inputs still work. 
1. sentence_a (string): A sentence 
2. sentence_b (string): A sentence that may or may not follow sentence_a

Returns: 
A float between 0 and 1 that represents how likely sentence_a follows sentence_b. 

#### Example 6.1:
```python
from happytransformer import HappyNextSentence
# --------------------------------------#
happy_ns = HappyNextSentence()
result = happy_ns.predict_next_sentence(
    "How old are you?",
    "I am 21 years old."
)
print(type(result))  # <class 'float'>
print(result)  # 0.9999918937683105
```

#### Example 6.2:
```python
from happytransformer import HappyNextSentence
# --------------------------------------#
happy_ns = HappyNextSentence()
result = happy_ns.predict_next_sentence(
    "How old are you?",
    "Queen's University is in Kingston Ontario Canada"
)
print(type(result))  # <class 'float'>
print(result)  # 0.00018497584096621722
```
