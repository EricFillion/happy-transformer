---
title: Settings
parent: Text-to-Text
nav_order: 2
layout: page
permalink: /text-to-text/settings/
---

# Text-to-Text Settings

By default a text generation algorithm called "greedy" is used.
This algorithm simply picks the most likely next word. 


A class called TTSettings() is used to control which algorithm is used and its settings. 
It is passed to the "args" parameter for HappyTextToText.generate_text(). 

```python
from happytransformer import TTSettings
```

TTSettings() contains the  fields shown in Table 1.0 

#### Table 7.0:  

| Parameter            |Default| Definition                                                                 |
|----------------------|-------|----------------------------------------------------------------------------|
| min_length           | 10    | Minimum number of generated tokens                                         |
| max_length           | 50    | Maximum number of generated tokens                                         |
| do_sample            | False | When True, picks words based on their conditional probability              |
| early_stopping       | False | When True, generation finishes if the EOS token is reached                 |
| num_beams            | 1     | Number of steps for each search path                                       |
| temperature          | 1.0   | How sensitive the algorithm is to selecting low probability options        |
| top_k                | 50    | How many potential answers are considered when performing sampling         | 
| top_p                | 1.0   | Min number of tokens are selected where their probabilities add up to top_p|
| no_repeat_ngram_size | 0     | The size of an n-gram that cannot occur more than once. (0=infinity)       |


#### Examples 7.2:  
 
 ```python
from happytransformer import HappyTextToText, TTSettings

#---------------------------------------------------
happy_tt = HappyTextToText("T5", "t5-small")

greedy_settings = TTSettings(no_repeat_ngram_size=2, max_length=20)
output_greedy = happy_tt.generate_text(
    "translate English to French: nlp is a field of artificial intelligence ",
    args=greedy_settings)

beam_settings = TTSettings(num_beams=5, max_length=20)
output_beam_search = happy_tt.generate_text(
    "translate English to French: nlp is a field of artificial intelligence ",
    args=beam_settings)

generic_sampling_settings = TTSettings(do_sample=True, top_k=0, temperature=0.7, max_length=20)
output_generic_sampling = happy_tt.generate_text(
    "translate English to French: nlp is a field of artificial intelligence ",
    args=generic_sampling_settings)

top_k_sampling_settings = TTSettings(do_sample=True, top_k=50, temperature=0.7, max_length=20)
output_top_k_sampling = happy_tt.generate_text(
    "translate English to French: nlp is a field of artificial intelligence ",
    args=top_k_sampling_settings)

top_p_sampling_settings = TTSettings(do_sample=True, top_k=0, top_p=0.8, temperature=0.7, max_length=20)
output_top_p_sampling = happy_tt.generate_text(
    "translate English to French: nlp is a field of artificial intelligence ",
    args=top_p_sampling_settings)

print("Greedy:", output_greedy.text)  # Greedy: nlp est un domaine de l'intelligence artificielle
print("Beam:", output_beam_search.text)  # Beam: nlp est un domaine de l'intelligence artificielle
print("Generic Sampling:", output_generic_sampling.text)  # Generic Sampling: nlp est un champ d'intelligence artificielle
print("Top-k Sampling:", output_top_k_sampling.text)  # Top-k Sampling: nlp est un domaine de l’intelligence artificielle
print("Top-p Sampling:", output_top_p_sampling.text)  # Top-p Sampling: nlp est un domaine de l'intelligence artificielle

```
## Tutorials 

[Top T5 Models ](https://www.vennify.ai/top-t5-transformer-models/)
[Grammar Correction](https://www.vennify.ai/grammar-correction-python/)
[Fine-tune a Grammar Correction Model](https://www.vennify.ai/fine-tune-grammar-correction/)

