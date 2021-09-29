---
title: Usage
parent: Text Classification
nav_order: 2
layout: page
permalink: /text-classification/usage/
---

## Text Classification Basic Usage

### classify_text()

Input: 
1. text (string): Text that will be classified 

Returns: 
An object with fields "label" and "score"

#### Example 2.1:
```python
from happytransformer import HappyTextClassification
# --------------------------------------#
happy_tc = HappyTextClassification(model_type="DISTILBERT",  model_name="distilbert-base-uncased-finetuned-sst-2-english")
result = happy_tc.classify_text("Great movie! 5/5")
print(type(result))  # <class 'happytransformer.happy_text_classification.TextClassificationResult'>
print(result)  # TextClassificationResult(label='POSITIVE', score=0.9998761415481567)
print(result.label)  # LABEL_1

```
