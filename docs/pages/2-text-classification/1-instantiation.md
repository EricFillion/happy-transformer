---
title: Text Classification
nav_order: 7
layout: page
permalink: /text-classification/
has_children: true
---

## Text Classification

Initialize a HappyTextClassification() object to perform text classification. 

This model assigns a label to a given text string. For example, you can train a model to 
detect if an email is spam based on its text. 


**Initialization Arguments:** 
1. model_type (string):  specify the model name in all caps, such as "ROBERTA" or "ALBERT"
2. model_name(string): below is a URL that contains potential models. The default is "distilbert-base-uncased"
       [MODELS](https://huggingface.co/models?filter=text-classification)
3. num_labels(int): The number of text categories. The default is 2 
4. use_auth_token (string): Specify the authentication token to 
   [load private](https://huggingface.co/transformers/model_sharing.html) models. 
5. from_tf (bool): Set to True if you want to convert a TensorFlow model to PyTorch model.


WARNING: If you try to load a pretrained model that has a different number of categories 
than num_labels, then you will get an error 

NOTE: "albert-base-v2", "bert-base-uncased" and "distilbert-base-uncased" do not have a predefined 
number of labels, so if you use these models you can set num_labels freely 


#### Example 2.0:
```python
from happytransformer import HappyTextClassification
# --------------------------------------#
happy_tc_distilbert = HappyTextClassification("DISTILBERT", "distilbert-base-uncased", num_labels=2)  # default 
happy_tc_albert = HappyTextClassification(model_type="ALBERT", model_name="albert-base-v2")
happy_tc_bert = HappyTextClassification("BERT", "bert-base-uncased")
happy_tc_roberta = HappyTextClassification("ROBERTA", "roberta-base")
happy_tc_private_roberta = HappyTextClassification("ROBERTA", "user-repo/roberta-base", use_auth_token="123abc")

```

## Tutorials 

[Text classification (training)](https://www.vennify.ai/train-text-classification-transformers/) 

[Text classification (hate speech detection)](https://youtu.be/jti2sPQYzeQ) 

[Text classification (sentiment analysis)](https://youtu.be/Ew72EAgM7FM)
