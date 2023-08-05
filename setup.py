# from distutils.core import setup
from setuptools import setup, find_packages

import pathlib

current_location = pathlib.Path(__file__).parent
readme = (current_location / "README.md").read_text()

setup(
    name = 'happytransformer',
    packages = find_packages(exclude=("tests",)),
    version = '3.0.0',
    license='Apache 2.0',
    description = "Happy Transformer makes it easy to fine-tune NLP Transformer models and use them for inference.",
    long_description= readme,
    long_description_content_type='text/markdown',
    author = "The Happy Transformer Development Team",
    author_email = 'happytransformer@gmail.com',
    url = 'https://github.com/EricFillion/happy-transformer',
    keywords = ['bert', 'roberta', 'ai', "transformer", "happy", "HappyTransformer",  "classification",  "nlp", "nlu", "natural", "language", "processing", "understanding"],

    install_requires=[
            'torch>=1.0',
            'tqdm>=4.43',
            'transformers>=4.30.1,<5.0.0',
            'datasets>=2.13.1,<3.0.0',
            'dataclasses; python_version < "3.7"',
            'sentencepiece',
            'protobuf',
            'accelerate>=0.20.1,<1.0.0',
            'tokenizers>=0.13.3,<1.0.0',
            'wandb'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

      ],
    )
