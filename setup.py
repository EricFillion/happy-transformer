from distutils.core import setup
setup(
  name = 'happytransformer',
  packages = ['happytransformer'],
  version = '1.0',
  license='MIT',
  description = 'Easily use XLNet, BERT and RoBERTa for masked word prediction and binary sequence classification',
  author = 'Eric Fillion, Umur Gokalp, Logan Roth and Xavier McMaster - Hubner',
  author_email = 'happytransformer@gmail.com',
  url = 'https://github.com/EricFillion/happy-transformer',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',
  keywords = ['BERT', 'roberta', 'xlnet', "word prediction", "masked", "transformer", "happy", "HappyTransformer", "binary", "sequence", "classification", "pytorch", "nlp", "nlu", "natural", "language", "processing", "understanding"],   # Keywords that define your package best


  install_requires=[
            'transformers',
            'logging',
            'numpy',
            'torch',
            'pandas',
            'multiprocessing',
            'tqdm',
            'numpy',
            'tqdm',
            'sklearn',

      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    # Check the code for other python version compatability
    'Programming Language :: Python :: 3.7',
    'Topic :: Software Development :: Build Tools',
    "Topic :: Text Processing :: Linguistic",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",

  ],
)