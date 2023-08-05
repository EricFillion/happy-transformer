[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![Downloads](https://pepy.tech/badge/happytransformer)](https://pepy.tech/project/happytransformer)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://happytransformer.com)
![PyPI](https://img.shields.io/pypi/v/happytransformer)
[![](https://github.com/EricFillion/happy-transformer/workflows/build/badge.svg)](https://github.com/EricFillion/happy-transformer/actions)

# Happy Transformer 
**Documentation and news: [happytransformer.com](http://happytransformer.com)**



Join our Discord server: [![Support Server](https://img.shields.io/discord/839263772312862740.svg?label=Discord&logo=Discord&colorB=7289da&style=?style=flat-square&logo=appveyor)](https://discord.gg/psVwe3wfTb)



![HappyTransformer](logo.png)

Happy Transformer makes it easy to fine-tune NLP Transformer models and use them for inference. 

## 3.0.0 
1. Deepspeed for training 
2. Apple's MPS for training and inference 
3. WandB to track training runs 
4. Data supplied for training is automatically split into portions for training and evaluating
5. Push models directly to Hugging Face's Model Hub

Read about the full 3.0.0 update including breaking changes [here](https://happytransformer.com/news/). 


## Tasks 
  
| Tasks                    | Inference | Training   |
|--------------------------|-----------|------------|
| Text Generation          | ✔         | ✔          |
| Text Classification      | ✔         | ✔          | 
| Word Prediction          | ✔         | ✔          |
| Question Answering       | ✔         | ✔          | 
| Text-to-Text             | ✔         | ✔          | 
| Next Sentence Prediction | ✔         |            | 
| Token Classification     | ✔         |            | 

## Quick Start
```sh
pip install happytransformer
```

```python

from happytransformer import HappyWordPrediction
#--------------------------------------#
happy_wp = HappyWordPrediction()  # default uses distilbert-base-uncased
result = happy_wp.predict_mask("I think therefore I [MASK]")
print(result)  # [WordPredictionResult(token='am', score=0.10172799974679947)]
print(result[0].token)  # am
```

## Maintainers
- [Eric Fillion](https://github.com/ericfillion)  Lead Maintainer
- [Ted Brownlow](https://github.com/ted537) Maintainer


## Tutorials 
[Text generation with training (GPT-Neo)](https://youtu.be/GzHJ3NUVtV4)

[Text classification (training)](https://www.vennify.ai/train-text-classification-transformers/) 

[Text classification (hate speech detection)](https://youtu.be/jti2sPQYzeQ) 

[Text classification (sentiment analysis)](https://youtu.be/Ew72EAgM7FM)

[Word prediction with training (DistilBERT, RoBERTa)](https://youtu.be/AWe0PHsPc_M)

[Top T5 Models ](https://www.vennify.ai/top-t5-transformer-models/)

[Grammar Correction](https://www.vennify.ai/grammar-correction-python/)

[Fine-tune a Grammar Correction Model](https://www.vennify.ai/fine-tune-grammar-correction/)
