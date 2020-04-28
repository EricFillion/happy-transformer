# from distutils.core import setup
from setuptools import setup

import pathlib

current_location = pathlib.Path(__file__).parent
readme = (current_location / "README.md").read_text()

setup(
    name = 'happytransformer',
    packages = ['happytransformer',],
    version = '1.1.3',
    license='Apache 2.0',
    description = "Happy Transformer is an API built on top of PyTorch's transformer library that makes it easy to utilize state-of-the-art NLP models.",
    long_description= readme,
    long_description_content_type='text/markdown',
    author = "The Happy Transformer Development Team",
    author_email = 'happytransformer@gmail.com',
    url = 'https://github.com/EricFillion/happy-transformer',
    keywords = ['bert', 'roberta', 'xlnet', "word",'prediction' "masked", "transformer", "happy", "HappyTransformer", "binary", "sequence", "classification", "pytorch", "nlp", "nlu", "natural", "language", "processing", "understanding"],


    install_requires=[
            'numpy',
            'torch',
            'pandas',
            'tqdm',
            'scikit_learn',
            'transformers',

      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

      ],
    )
