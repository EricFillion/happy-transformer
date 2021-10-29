---
title: Text-to-Text
nav_order: 12
layout: page
permalink: /text-to-text/
has_children: true
---

## Text-to-Text

Initialize a HappyTextToText() object to perform text-to-text generation

**Initialization Arguments:**
 1. model_type (string): Specify the model name in all caps, such as "T5" or "BART" 
 2. model_name(string): below are  URLs that contains potential models: 
       [standard models](https://huggingface.co/models?pipeline_tag=text2text-generation) and [translation models](https://huggingface.co/models?pipeline_tag=translation)
 3. use_auth_token (string): Specify the authentication token to 
       [load private](https://huggingface.co/transformers/model_sharing.html) models. 


#### Example 7.0:
```python
from happytransformer import HappyTextToText
# --------------------------------------#
happy_tt = HappyTextToText("T5", "t5-small")  # default
happy_tt_private = HappyTextToText("T5", "user-repo/t5-small", use_auth_token="123abc")  # default

```
