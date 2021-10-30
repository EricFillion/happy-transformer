---
title: Usage
parent: Text-to-Text
nav_order: 1
layout: page
permalink: /text-to-text/usage/
---

## Text-to-Text Basic Usage
### generate_text()
The method generate_text() contains 2 arguments:
1. text (string): The text prompt for the model. 
2. args (TTSettings): See this [webpage](/text-to-text/settings/) for more information


Returns: 
An object with a single field called "text"


#### Example 7.1:
```python

from happytransformer import HappyTextToText, TTSettings
#--------------------------------------#
happy_tt = HappyTextToText()  # default uses t5-small
top_p_sampling_settings = TTSettings(do_sample=True, top_k=0, top_p=0.8, temperature=0.7,  min_length=20, max_length=20, early_stopping=True)
result = happy_tt.generate_text("translate English to French: nlp is a field of artificial intelligence", args=top_p_sampling_settings)
print(result)  # TextToTextResult(text="nlp est un domaine de l'intelligence artificielle...")
print(result.text)  # nlp est un domaine de l’intelligence artificielle. n

```
## Tutorials

[Top T5 Models ](https://www.vennify.ai/top-t5-transformer-models/)
[Grammar Correction](https://www.vennify.ai/grammar-correction-python/)
[Fine-tune a Grammar Correction Model](https://www.vennify.ai/fine-tune-grammar-correction/)

