---
layout: page
title: Pushing to Hugging Face's Hub
permalink: /push/
nav_order: 15
---
## Pushing To Hugging Face's Model Hub


All Happy Transformer objects can be pushed to Hugging Face's [Model Hub](https://huggingface.co/).

First log into Hugging Face 

```bash
huggingface-cli login
```

```python
from happytransformer import HappyGeneration
# ---------------------------------------------------------
happy_gen_1 = HappyGeneration(model_type="GPT-NEO", model_name="EleutherAI/gpt-neo-125M")
repo_name = "EricFillion/example"
happy_gen_1.push(repo_name, private=True)

# Be sure to set use_auth_token to True
happy_gen_2 = HappyGeneration("GPT-2", "EricFillion/example", use_auth_token=True)


```



