from dataclasses import dataclass
from happytransformer.happy_trainer import HappyTrainer
from happytransformer.tt.default_args import ARGS_TT_TRAIN, ARGS_TT_EVAL, ARGS_TT_TEST

@dataclass
class TTTrainArgs:
    learning_rate: float = ARGS_TT_TRAIN["learning_rate"]
    num_train_epochs: int = ARGS_TT_TRAIN["num_train_epochs"]
    weight_decay: float = ARGS_TT_TRAIN["weight_decay"]
    adam_beta1: float = ARGS_TT_TRAIN["adam_beta1"]
    adam_beta2: float = ARGS_TT_TRAIN["adam_beta2"]
    adam_epsilon: float = ARGS_TT_TRAIN["adam_epsilon"]
    max_grad_norm:  float = ARGS_TT_TRAIN["max_grad_norm"]
    save_preprocessed_data: bool = ARGS_TT_TRAIN["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_TT_TRAIN["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_TT_TRAIN["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_TT_TRAIN["load_preprocessed_data_path"]
    batch_size: int = ARGS_TT_TRAIN["batch_size"]


@dataclass
class TTEvalArgs:
    batch_size: int = ARGS_TT_EVAL["batch_size"]
    save_preprocessed_data: bool = ARGS_TT_EVAL["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_TT_EVAL["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_TT_EVAL["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_TT_EVAL["load_preprocessed_data_path"]


@dataclass
class TTTestArgs:
    save_preprocessed_data: bool = ARGS_TT_TEST["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_TT_TEST["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_TT_TEST["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_TT_TEST["load_preprocessed_data_path"]



class TTTrainer(HappyTrainer):

    def train(self, input_filepath, args=TTTrainArgs):
        raise NotImplementedError()

    def eval(self, input_filepath, args=TTEvalArgs):
        raise NotImplementedError()

    def test(self, input_filepath, solve, args=TTTestArgs):
        raise NotImplementedError()