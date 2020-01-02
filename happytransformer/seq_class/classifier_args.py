classifier_args = {
    # Basic fine tuning parameters
    'learning_rate': 1e-5,
    'num_epochs': 2,
    'batch_size': 8,

    # More advanced fine tuning parameters
    'max_seq_length': 128,  # The maximum tokens allowed in each input. Max value = 512. Increasing it significantly increases memory usage
    'adam_epsilon': 1e-5,
    'gradient_accumulation_steps': 1,
    'weight_decay': 0,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    # More modes will become available in future releases
    'task_mode': 'binary',

}