---
title: Usage
parent: Word Prediction
nav_order: 1
layout: page
permalink: /word-prediction/usage/
---

## Word Prediction Basic Usage 
### predict_mask()
The method predict_masks() contains 3 arguments: 
1. text (string): a body of text that contains a single masked token 
2. targets (list of strings): a list of potential answers. All other answers will be ignored 
3. top_k (int): the number of results that will be returned 

Returns: 
A list of objects with fields "token" and "score"

Note: if targets are provided, then top_k will be ignored and a score for each target will be returned. 

#### Example 4.1:
```python

from happytransformer import HappyWordPrediction
#--------------------------------------#
happy_wp = HappyWordPrediction()  # default uses distilbert-base-uncased
result = happy_wp.predict_mask("I think therefore I [MASK]")
print(type(result))  # <class 'list'>
print(result)  # [WordPredictionResult(token='am', score=0.10172799974679947)]
print(type(result[0]))  # <class 'happytransformer.happy_word_prediction.WordPredictionResult'>
print(result[0])  # [WordPredictionResult(token='am', score=0.10172799974679947)]
print(result[0].token)  # am
print(result[0].score)  # 0.10172799974679947
    

```

#### Example 4.2:
```python

from happytransformer import HappyWordPrediction
#--------------------------------------#
happy_wp = HappyWordPrediction()
result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", top_k=2)
print(result)  # [WordPredictionResult(token='health', score=0.1280556619167328), WordPredictionResult(token='science', score=0.07976455241441727)]
print(result[1]) # WordPredictionResult(token='science', score=0.07976455241441727)
print(result[1].token) # science

```

#### Example 4.3:
```python
from happytransformer import HappyWordPrediction
#--------------------------------------#
happy_wp = HappyWordPrediction()
targets = ["technology", "healthcare"]
result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", targets=targets, top_k=2)
print(result)  # [WordPredictionResult(token='healthcare', score=0.07380751520395279), WordPredictionResult(token='technology', score=0.009395276196300983)]
print(result[1])  # WordPredictionResult(token='technology', score=0.009395276196300983)
print(result[1].token)  # technology


```