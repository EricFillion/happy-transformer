---
layout: page
title: Pushing to Hugging Face's Hub
permalink: /push-to-hub/
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
happy_gen = HappyGeneration(model_type="GPT-NEO", model_name="EleutherAI/gpt-neo-125M")
repo_name = "ericfillion/example"
happy_gen.push_to_hub(repo_name, private=True)

```



