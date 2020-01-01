classifier_args = {
    # Basic fine tuning parameters
    'learning_rate': 4e-5,
    'num_train_epochs': 1,
    'batch_size': 2,

    'max_seq_length': 128,
    'adam_epsilon': 1e-8,

    'gradient_accumulation_steps': 1,
    'weight_decay': 0,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,


    'fp16': False,
    'fp16_opt_level': 'O1',

    'task_mode': 'binary',

}