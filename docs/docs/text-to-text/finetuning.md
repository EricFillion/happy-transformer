
HappyTextToText contains two methods for training 

- train(): fine-tune the model to convert a standalone text to another standalone piece of text 

- eval(): determines how well the model performs 

### train()

inputs: 
1. input_filepath (string): a path file to a csv file as shown in table 7.1

2. args (TTTrainArgs): a dataclass with the same fields types as shown in Table 7.2. 

3. eval_filepath (string): By default, an evaluating dataset will be generated from the supplied training data. But, you may provide a filepath to a CSV file as described for input_filepath to use standalone evaluating data. 


#### Table 7.1
Contains two columns with the following header values: input and target

| input                         |target               |
|-------------------------------|---------------------|
| grammar: I has poor grammars  | I have poor grammar |
| grammar: I wants too plays    | I want to play      |


#### Table 7.2

| Parameter                | Default             |
|--------------------------|---------------------|
| learning_rate            | 5e-5                |
| num_train_epochs         | 1                   |
| batch_size               | 1                   |
| preprocessing_processes  | 1                   |
| save_path                | ""                  |
| load_path                | ""                  |
| max_input_length         | None                |
| max_output_length        | None                |
| fp16                     | False               |
| eval_ratio               | 0.1                 |
| save_steps               | 0.0                 |
| eval_steps               | 0.1                 |
| logging_steps            | 0.1                 |
| output_dir               | "happy_transformer" |


Information about the learning parameters can be found [here](/learning-parameters/)


preprocessing_processes: Number of processes used for preprocessing. We recommend 1-4. 

max_input_length: The maximum number of tokens for the input. The rest get truncated. By default the maximum number of tokens the model can handle is used. 

max_output_length: Ditto, except for the output. 


#### Example 7.3:
```python
from happytransformer import HappyTextToText, TTTrainArgs
# --------------------------------------#
happy_tt = HappyTextToText()
args = TTTrainArgs(num_train_epochs=1) 
happy_tt.train("../../data/tt/train-eval-grammar.csv", args=args)
```

### eval()
Input:

1. input_filepath (string): a path file to a csv file with the same format as described for the training data in table 7.1

2. args (TTEvalArgs): a dataclass with the same fields shown in Table 7.3

#### Table 7.3

| Parameter                     |Default|
|-------------------------------|-------|
| preprocessing_processes       | 1     |
| max_input_length              | 1024  |
| max_output_length             | 1024  |

See Table 7.1 for more information 


Output: An object with a single field called "loss"

#### Example 1.4
```python
from happytransformer import HappyTextToText, TTEvalArgs
# --------------------------------------#
happy_tt = HappyTextToText()
args = TTEvalArgs(preprocessing_processes=1)
result = happy_tt.eval("../../data/tt/train-eval-grammar.csv", args=args)
print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
print(result)  # EvalResult(loss=3.2277376651763916)
print(result.loss)  # 3.2277376651763916

```
## Tutorials 

[Top T5 Models ](https://www.vennify.ai/top-t5-transformer-models/)
[Grammar Correction](https://www.vennify.ai/grammar-correction-python/)
[Fine-tune a Grammar Correction Model](https://www.vennify.ai/fine-tune-grammar-correction/)
