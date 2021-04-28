ARGS_WP_TRAIN = {
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
    'line_by_line': False,

    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",

    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
}


ARGS_WP_EVAl = {
    'preprocessing_processes': 1,
    'mlm_probability': 0.15,
    'line_by_line': False,

    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",

    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",

}

ARGS_WP_TEST = {
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",

    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",

}
