---
title: Question Answering
nav_order: 8
layout: page
permalink: /question-answering/
has_children: true

---

## Question Answering 
Initialize a HappyQuestionAnswering() object to perform question answering. 

This model answers a question given a body of that's text relevant to the questions. 

The outputted answer is always a text-span with the provided information. 

**Initialization Arguments:**
1. model_type (string): specify the model name in all caps, such as "ROBERTA" or "ALBERT"
2. model_name(string): below is a URL that contains potential models. 
   [MODELS](https://huggingface.co/models?filter=question-answering)
3. use_auth_token (string): Specify the authentication token to 
   [load private](https://huggingface.co/transformers/model_sharing.html) models. 
 4. from_tf (bool): Set to True if you want to convert a TensorFlow model to PyTorch model.

We recommend using "HappyQuestionAnswering("ALBERT", "mfeb/albert-xxlarge-v2-squad2")" for the best performance 


#### Example 3.0:
```python
from happytransformer import HappyQuestionAnswering
# --------------------------------------#
happy_qa_distilbert = HappyQuestionAnswering("DISTILBERT", "distilbert-base-cased-distilled-squad")  # default
happy_qa_albert = HappyQuestionAnswering("ALBERT", "mfeb/albert-xxlarge-v2-squad2")
# good model when using with limited hardware 
happy_qa_bert = HappyQuestionAnswering("BERT", "mrm8488/bert-tiny-5-finetuned-squadv2")
happy_qa_roberta = HappyQuestionAnswering("ROBERTA", "deepset/roberta-base-squad2")
happy_qa__private_roberta = HappyQuestionAnswering("ROBERTA", "user-repo/roberta-base-squad2", use_auth_token="123abc")

```
