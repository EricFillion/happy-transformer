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

| Parameter            |Default| Definition                                                          |
|----------------------|-------| --------------------------------------------------------------------|
| do_sample            | False | When True, picks words based on their conditional probability       |
| early_stopping       | False | When True, generation finishes if the EOS token is reached          |
| num_beams            | 1     | Number of steps for each search path                                |
| temperature          | 1     | How sensitive the algorithm is to selecting low probability options |
| top_k                | 50    | How many potential answers are considered when performing sampling  |
| no_repeat_ngram_size | 0     | The maximum length of an n-gram within the generated text.          |
   
#### Examples 1.2:  
 
 ```python
from happytransformer import HappyGeneration, GENSettings

#---------------------------------------------------
    happy_gen = HappyGeneration()

    greedy_settings = GENSettings(no_repeat_ngram_size=2)
    output_greedy = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=greedy_settings, min_length=5, max_length=5)

    beam_settings = GENSettings(early_stopping=True, num_beams=5)
    output_beam_search = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=beam_settings, min_length=5, max_length=5)

    generic_sampling_settings = GENSettings(do_sample=True, early_stopping=False, top_k=0, temperature=0.7)
    output_generic_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=generic_sampling_settings, min_length=5, max_length=5)

    top_k_sampling_settings = GENSettings(do_sample=True, early_stopping=False, top_k=50, temperature=0.7)
    output_top_k_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=top_k_sampling_settings, min_length=5, max_length=5)
```

