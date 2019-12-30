classifier_args = {
    'model_type':  '',
    'model_name': '',
    'task_mode': 'binary',
    'task': "idle",
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'train_batch_size': 2, # was 8
    'eval_batch_size': 2, # was 8

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'evaluate_during_training': True,
    'reprocess_input_data': True,
    'gpu_support': 'cpu',
}