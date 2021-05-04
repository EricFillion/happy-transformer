ARGS_GEN_TRAIN = {
    'learning_rate': 5e-5,
    'num_train_epochs': 3.0,
    'batch_size': 1,
    'weight_decay': 0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'max_grad_norm':  1.0,
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
    'preprocessing_processes': 1,
    'mlm_probability': 0.15,
}


ARGS_GEN_EVAl = {
    'batch_size': 1,
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
    'preprocessing_processes': 1,
    'mlm_probability': 0.15,

}


ARGS_GEN_TEST = {
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",

}
