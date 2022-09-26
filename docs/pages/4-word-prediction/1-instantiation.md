---
title: Word Prediction
nav_order: 9
layout: page
permalink: /word-prediction/
has_children: true

---

## Word Prediction

See [Medium article](https://medium.com/vennify-ai/masked-word-prediction-using-transformer-models-e7b0385f909) for a more in-depth explanation 

Initialize a HappyWordPrediction object to perform word prediction. 

**Initialization Arguments:**
 1. model_type (string): Specify the model name in all caps, such as "ROBERTA" or "ALBERT" 
 2. model_name(string): below is a URL that contains potential models: 
       [MODELS](https://huggingface.co/models?filter=masked-lm)
 3. use_auth_token (string): Specify the authentication token to 
       [load private](https://huggingface.co/transformers/model_sharing.html) models. 
 4. from_tf (bool): Set to True if you want to convert a TensorFlow model to PyTorch model.

Note: For all Transformers, the masked token is **"[MASK]"**

We recommend using "HappyWordPrediction("ALBERT", "albert-xxlarge-v2")" for the best performance 


#### Example 4.0:
```python
from happytransformer import HappyWordPrediction
# --------------------------------------#
happy_wp_distilbert = HappyWordPrediction("DISTILBERT", "distilbert-base-uncased")  # default
happy_wp_albert = HappyWordPrediction("ALBERT", "albert-base-v2")
happy_wp_bert = HappyWordPrediction("BERT", "bert-base-uncased")
happy_wp_roberta = HappyWordPrediction("ROBERTA", "roberta-base")
happy_wp_private_roberta = HappyWordPrediction("ROBERTA", "user-repo/roberta-base", use_auth_token="123abc")

```
## Tutorials 

[Word prediction with training (DistilBERT, RoBERTa)](https://youtu.be/AWe0PHsPc_M)
