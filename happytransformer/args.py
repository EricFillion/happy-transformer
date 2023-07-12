from dataclasses import dataclass

@dataclass
class TrainArgs:
    output_dir: str = "happy_transformer/"
    learning_rate: float = 5e-5
    num_train_epochs: int = 1.0
    batch_size: int = 1
    gas: int = 1
    weight_decay: float = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm:  float = 1.0
    fp16: bool = False
    eval_ratio: float = 0.1  #  if eval_filepath is not provided a portion of the training data will be used for evaluating.

    save_steps: float = 0.0 #  if 0 no saving will be done
    eval_steps: float = 2  # if 0 no evaluating will be done
    logging_steps: float = 0.0   # if 0 no saving will be done

    load_preprocessed_data: bool = False
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data_path: str = ""
    preprocessing_processes: int = 1
    report_to: tuple = ()
    deepspeed: str = ""

    # Currently used to create a project and run ID for wandb
    project_name: str = "happy-transformer"
    run_name: str = "test-run"

@dataclass
class EvalArgs:
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data: bool = False
    load_preprocessed_data_path: str = ""
    batch_size: int = 1
    preprocessing_processes: int = 1

@dataclass
class TestArgs:
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data: bool = False
    load_preprocessed_data_path: str = ""

# GEN
@dataclass
class GENTrainArgs(TrainArgs):
    pass

@dataclass
class GENEvalArgs(EvalArgs):
    pass

# QA
@dataclass
class QATrainArgs(TrainArgs):
    pass

@dataclass
class QAEvalArgs(EvalArgs):
    pass

@dataclass
class QATestArgs(TestArgs):
    pass

# TC
@dataclass
class TCTrainArgs(TrainArgs):
    pass

@dataclass
class TCEvalArgs(EvalArgs):
    pass

@dataclass
class TCTestArgs(TestArgs):
    pass

# WP
@dataclass
class WPTrainArgs(TrainArgs):
    mlm_probability: float = 0.1
    line_by_line: bool = False

@dataclass
class WPEvalArgs(EvalArgs):
    mlm_probability: float = 0.1
    line_by_line: bool = False

@dataclass
class WPTestArgs(TestArgs):
    pass

# TT
@dataclass
class TTTrainArgs(TrainArgs):
    max_input_length: int = 1024
    max_output_length: int = 1024

@dataclass
class TTEvalArgs(EvalArgs):
    max_input_length: int = 1024
    max_output_length: int = 1024

@dataclass
class TTTestArgs(TestArgs):
    pass
