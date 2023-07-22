---
layout: page
title: News
permalink: /news/
nav_order: 4

---
## News: 

### July 22nd, 2023 

**Version 2.5.0!**
- Deepspeed is now supported for fine-tuning. 
- Apple's MPS backend is now automatically used for both training and fine-tuning if detected. 
- Evaluating data is now used during fine-tuning to track the fine-tuning progress. 
- WandB can now be used to log the results from fine-tuning. 
- Apple's MPS chips can be used for inference and fine-tuning.
- CSV files are supported for training/evaluating text generation and word prediction models. This makes it easy to isolate cases. 
- Push models to Hugging Face's Hub with one command. 

Breaking changes:
- Preprocesses data is now saved in the Hugging Face's Dataset format rather than in JSON format.
- Dictionary argument inputs for training and evaluating are no longer supported 

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