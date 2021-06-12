# from distutils.core import setup
from setuptools import setup, find_packages

import pathlib

current_location = pathlib.Path(__file__).parent
readme = (current_location / "README.md").read_text()

setup(
    name = 'happytransformer',
    packages = find_packages(),
    version = '2.2.3',
    license='Apache 2.0',
    description = "Happy Transformer is an API built on top of Hugging Face's Transformer library that makes it easy to utilize state-of-the-art NLP models.",
    long_description= readme,
    long_description_content_type='text/markdown',
    author = "The Happy Transformer Development Team",
    author_email = 'happytransformer@gmail.com',
    url = 'https://github.com/EricFillion/happy-transformer',
    keywords = ['bert', 'roberta', 'xlnet', "transformer", "happy", "HappyTransformer",  "classification",  "nlp", "nlu", "natural", "language", "processing", "understanding"],

    install_requires=[
            'torch>=1.0',
            'tqdm>=4.27',
            'transformers>=4.4.0',
            'datasets>=1.6.0',
            'dataclasses; python_version < "3.7"',
            'sentencepiece',
            'protobuf'

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
