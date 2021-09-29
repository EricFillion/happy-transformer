---
title: Settings
parent: Text Generation
nav_order: 2
layout: page
permalink: /text-generation/settings/
---

# Text Generation Settings

By default a text generation algorithm called "greedy" is used.
This algorithm simply picks the most likely next word. 
However, there are more sophisticated ways to perform next generation as described in 
this [article](https://huggingface.co/blog/how-to-generate) by Hugging Face. 

A class called GENSettings() is used to control which algorithm is used and its settings. 
It is passed to the "args" parameter for HappyGeneration.generate_text(). 

```python
from happytransformer import GENSettings
```

GENSettings() contains the  fields shown in Table 1.0 

#### Table 1.0:  

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
| bad_words            | None  | List of words/phrases that cannot be generated.                            | 


#### Example 1.2:  
 
 ```python
from happytransformer import HappyGeneration, GENSettings

#---------------------------------------------------
happy_gen = HappyGeneration()

greedy_settings = GENSettings(no_repeat_ngram_size=2,  max_length=10)
output_greedy = happy_gen.generate_text(
    "Artificial intelligence is ",
    args=greedy_settings)

beam_settings = GENSettings(num_beams=5,  max_length=10)
output_beam_search = happy_gen.generate_text(
    "Artificial intelligence is ",
    args=beam_settings)

generic_sampling_settings = GENSettings(do_sample=True, top_k=0, temperature=0.7,  max_length=10)
output_generic_sampling = happy_gen.generate_text(
    "Artificial intelligence is ",
    args=generic_sampling_settings)

top_k_sampling_settings = GENSettings(do_sample=True, top_k=50, temperature=0.7,  max_length=10)
output_top_k_sampling = happy_gen.generate_text(
    "Artificial intelligence is ",
    args=top_k_sampling_settings)

top_p_sampling_settings = GENSettings(do_sample=True, top_k=0, top_p=0.8, temperature=0.7,  max_length=10)
output_top_p_sampling = happy_gen.generate_text(
    "Artificial intelligence is ",
    args=top_p_sampling_settings)

bad_words_settings = GENSettings(bad_words = ["new form", "social"])
output_bad_words = happy_gen.generate_text(
    "Artificial intelligence is ",
    args=bad_words_settings)
    
print("Greedy:", output_greedy.text)  # a new field of research that has been gaining
print("Beam:", output_beam_search.text) # one of the most promising areas of research in
print("Generic Sampling:", output_generic_sampling.text)  #  an area of highly promising research, and a
print("Top-k Sampling:", output_top_k_sampling.text)  # a new form of social engineering. In this
print("Top-p Sampling:", output_top_p_sampling.text)  # a new form of social engineering. In this
print("Bad Words:", output_bad_words.text) # a technology that enables us to help people deal
```

