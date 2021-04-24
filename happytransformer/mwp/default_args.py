ARGS_MWP_TRAIN = {
    # training args
    'learning_rate': 5e-5,
    'weight_decay': 0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'max_grad_norm':  1.0,
    'num_train_epochs': 3.0,
    'preprocessing_processes': 1,
    'mlm_probability': 0.15,
    'line-by-line': False
}


ARGS_MWP_EVAl = {
    'preprocessing_processes': 1,
    'mlm_probability': 0.15,
    'line-by-line': False

}

ARGS_MWP_TEST = {}