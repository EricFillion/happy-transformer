[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![Downloads](https://pepy.tech/badge/happytransformer)](https://pepy.tech/project/happytransformer)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://happytransformer.com)
![PyPI](https://img.shields.io/pypi/v/happytransformer)
[![](https://github.com/EricFillion/happy-transformer/workflows/build/badge.svg)](https://github.com/EricFillion/happy-transformer/actions)

# Happy Transformer 
**Documentation and news: [happytransformer.com](http://happytransformer.com)**


Join our brand new Discord server: [![Support Server](https://img.shields.io/discord/839263772312862740.svg?label=Discord&logo=Discord&colorB=7289da&style=?style=flat-square&logo=appveyor)](https://discord.gg/psVwe3wfTb)



![HappyTransformer](logo.png)

Happy Transformer is an package built on top of [Hugging Face's transformer library](https://huggingface.co/transformers/) that makes it easy to utilize state-of-the-art NLP models. 

## Features 
  
| Public Methods                     | Basic Usage  | Training   |
|------------------------------------|--------------|------------|
| Word Prediction                    | ✔            | ✔          |
| Text Generation                    | ✔            | ✔          |
| Text Classification                | ✔            | ✔          | 
| Question Answering                 | ✔            | ✔          | 
| Next Sentence Prediction           | ✔            |            | 
| Token Classification               | ✔            |            | 

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


## Maintainers
- [Eric Fillion](https://github.com/ericfillion)  Lead Maintainer
- [Ted Brownlow](https://github.com/ted537) Maintainer
