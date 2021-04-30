## Word Prediction Finetuning

HappyWordPrediction contains three methods for training 
- train(): fine-tune the model to understand a body of text better
- eval(): determine how well the model performs 

### train()

inputs: 
1. input_filepath (string): a path file to a text file that contains nothing but text to train the model with
2. args (dictionary): a dictionary with the same keys and value types as shown below. 

```python
ARGS_WP_TRAIN= {
    #-------------------------------------------
    # learning parameters: More information can be found on Hugging Face's website below 
    'learning_rate': 5e-5,
    'weight_decay': 0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'max_grad_norm':  1.0,
    'num_train_epochs': 3.0,
    #-------------------------------------------
    # Pre-processing parameters
    # See below for descriptions of each
    
    'preprocessing_processes': 1, 
    'mlm_probability': 0.15,
    'line-by-line': False
}
```
[Hugging Face Learning Parameters](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)

preprocessing_processes: Number of processes to use for pre-processing. We recommend 1-4. 
mlm_probability: The probability of masking a token.
line-by-line: If False, training data is concatenated and then divided into sections that are the length of the model's input size, other than the last input which may be shorter. 
              If True, each input contains the text from a single line within the training data. The text may be truncated if the line is too long (eg BERT's max input size is 512 tokens). 



#### Example 1.4:
```python
    from happytransformer import HappyWordPrediction, ARGS_WP_TRAIN
    # --------------------------------------#
    
    happy_wp = HappyWordPrediction()
    
    args = ARGS_WP_TRAIN # default values
    args["num_train_epochs"] = 1 # change number of epochs from 3 to 1
    happy_wp.train("../../data/wp/train-eval.txt", args=args)
```

### eval()
Input:
1. input_filepath (string): a path file to a csv file as described in table 2.1
2. args (dictionary): a dictionary with the same keys and value types as shown below. 
```python
ARGS_WP_EVAL = {
    # These keys are described under ARGS_WP_TRAIN
    'preprocessing_processes': 1,
    'mlm_probability': 0.15,
    'line-by-line': False
}
```

Output: An object with the field "loss"

#### Example 1.5
```python
    from happytransformer import HappyWordPrediction, ARGS_WP_EVAl
    # --------------------------------------#
    happy_wp = HappyWordPrediction()  
    args = ARGS_WP_EVAl
    args['preprocessing_processes'] = 2 # changed from 1 to 2
    result = happy_wp.eval("../../data/wp/train-eval.txt")
    print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
    print(result)  # EvalResult(eval_loss=0.459536075592041)
    print(result.loss)  # 0.459536075592041
```