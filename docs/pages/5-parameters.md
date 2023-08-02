---
layout: page
title: Learning Parameters 
permalink: /learning-parameters/
nav_order: 12

---

## Learning Parameters

learning_rate: How much the model's weights are adjusted per step. Too low and the model will take a long time to learn or get stuck in a suboptimal solution. Too high can cause can divergent behaviors.    

num_train_epochs: The number of times the training data is iterated over. 

weight_decay: A type of regularization. It prevents weights from getting too large. Thus, preventing overfitting. 

batch_size: Number of training examples used per iteration 

gradient_accumulation_steps: The number of batches that occur before the model weights are updated. 

fp16: If true, enables half precision training which saves space by using 16 bits instead of 32 to store the model's weights. Only available when CUDA/a a GPU is being used.   

eval_ratio: The ratio of data supplied to input_filepath that will be used for evaluating. If eval_filepath is supplied this argument is ignored and input_filepath is used only as train data. 

save_steps: Ratio of total train step before saving occurs. 

eval_steps: Ratio of total train step before evaluating occurs. 

logging_steps: Ratio of total train step before logging occurs. 

output_dir: An output directory where models will be saved to if save_steps is enabled. Other features that leverage this directory may be added in the future. 