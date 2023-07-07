# GEN

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
    'fp16': False
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



# QA
ARGS_QA_TRAIN = {
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
    'fp16': False
}


ARGS_QA_EVAl = {
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
    'batch_size': 1,
}

ARGS_QA_TEST = {
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
}

# SP
ARGS_SP_TRAIN = {
    'learning_rate': 5e-5,
    'weight_decay': 0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'max_grad_norm':  1.0,
    'num_train_epochs': 3.0,
}

# TC

ARGS_TC_TRAIN = {
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
    'fp16': False

}


ARGS_TC_EVAL = {
    'batch_size': 1,
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
}

ARGS_TC_TEST = {
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
}

# TOC
ARGS_TOC_TRAIN = {}

ARGS_TOC_EVAl = {}

ARGS_TOC_TEST = {}


# WP
ARGS_WP_TRAIN = {
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
    'line_by_line': False,
    'fp16': False
}


ARGS_WP_EVAl = {
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",
    'preprocessing_processes': 1,
    'mlm_probability': 0.15,
    'line_by_line': False,
    'batch_size': 1,

}

ARGS_WP_TEST = {
    'save_preprocessed_data': False,
    'save_preprocessed_data_path': "",
    'load_preprocessed_data': False,
    'load_preprocessed_data_path': "",

}
