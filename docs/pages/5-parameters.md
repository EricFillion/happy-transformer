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

adam_beta1: The beta1 parameter for the Adam with weight decay optimizer. 

adam_beta2: The beta2 parameter for the Adam with weight decay optimizer.

adam_epsilon: The epsilon parameter for the Adam with weight decay optimizer.

max_grad_norm: Used to prevent exploding gradients. Prevents the derivatives of the loss function from exceed the absolute value of "max_grad_norm". 

batch_size: Number of training examples used per iteration 

fp16: If true, enables half precision training which saves space by using 16 bits instead of 32 to store the model's weights. Only available when CUDA/a a GPU is being used.   