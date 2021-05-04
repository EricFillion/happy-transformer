---
title: Usage
parent: Text Generation
nav_order: 1
layout: page
permalink: /text-generation/usage/
---

## Text Generation Basic Usage 
### generate_text()
The method predict_masks() contains 4 arguments: 
1. text (string): The text prompt for the model -- it will try to continue the text
2. settings (GENSettings): See this [webpage](/text-generation/settings/) for more information 


Returns: 
An object with a single field called "text"


#### Example 1.1:
```python

from happytransformer import HappyGeneration, GENSettings
#--------------------------------------#
    happy_wp = HappyGeneration()  # default uses distilbert-base-uncased
    args = GENSettings(max_length=15)
    result = happy_wp.generate_text("artificial intelligence is ", args=args)    
    print(result)  # GenerationResult(text='\xa0a new field of research that has been gaining momentum in recent years.')
    print(result.text)  #  a new field of research that has been gaining momentum in recent years.

```

