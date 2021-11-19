---
layout: page
title: News
permalink: /news/
nav_order: 4

---
## News: 

### November 19th, 2021
**Introducing Version 2.4.0!**
- We added the ability to enable half-precision training, which decreases train time and memory consumption.
Just simply set the "fp16" training argument to True while using CUDA/ a GPU.   
- We also set the character encoding format to utf-8 for HappyTextClassification and HappyQuestionAnswering. Before it would change based on your system.  

### October 29th, 2021
- You can now use private models from Hugging Face's Model Hub

### August 14th, 2021
**Introducing Version 2.3.0!**

New Features: 
- Text-to-text fine-tuning is now available!

### May 4th, 2021
**Introducing Version 2.2.0!**

New Features: 
- Text generation with training 
- Word prediction training   
- Saving/loading models 
- Saving/loading preprocessed data  
- You can now change the batch size when training and evaluating  
- Dataclasses can now be used for all finetuning "arg" parameters 


### March 1st, 2021
**Introducing Version 2.1.0!**
You can now use any model type available on [Hugging Face's model distribution network](https://huggingface.co/models) for the implemented features. 
This includes BERT, ROBERTA, ALBERT XLNET and more. 

You can also now perform token classification 


### January 12, 2021
**Introducing Version 2.0.0!**

We fully redesigned Happy Transformer from the ground up. 

New Features: 
- Question answering training 
- Multi label text classification training
- Single predictions for text classification 

Deprecated Features: 
- Masked word prediction training
- Masked word prediction with multiple masks 

Breaking changes: 
- Everything

Happy Transformer have been redesigned to promote scalability. 
Now it's easier than ever to add new models and features, and we encourage you
to create PRs to contribute to the project. 