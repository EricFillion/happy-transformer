from dataclasses import dataclass
from typing import Union

@dataclass
class TrainArgs:
    output_dir: str = "happy_transformer/"
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    batch_size: int = 1
    weight_decay: float = 0
    fp16: bool = False
    eval_ratio: float = 0.1  #  if eval_filepath is not provided a portion of the training data will be used for evaluating.

    save_steps: float = 0.0 #  if 0 no saving will be done
    eval_steps: float = 0.1  # if 0 no evaluating will be done
    logging_steps: float = 0.1   # if 0 no saving will be done

    save_path:  Union[bool, str] = False
    load_path:  Union[bool, str] = False

    report_to: tuple = ()
    deepspeed: Union[bool, str] = False

    # Currently used to create a project and run ID for wandb
    project_name: str = "happy-transformer"
    run_name: str = "test"

@dataclass
class EvalArgs:
    batch_size: int = 1
    deepspeed: Union[bool, str] = False

    save_path:  Union[bool, str] = False
    load_path:  Union[bool, str] = False

@dataclass
class TestArgs:
   pass

# GEN
@dataclass
class GENTrainArgs(TrainArgs):
    max_length: Union[None, int] = None

@dataclass
class GENEvalArgs(EvalArgs):
    max_length: Union[None, int] = None

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
    max_length: Union[None, int] = None

@dataclass
class WPEvalArgs(EvalArgs):
    mlm_probability: float = 0.1
    line_by_line: bool = False
    max_length: Union[None, int] = None

@dataclass
class WPTestArgs(TestArgs):
    pass

# TT
@dataclass
class TTTrainArgs(TrainArgs):
    max_input_length: Union[None, int] = None
    max_output_length: Union[None, int] = None

@dataclass
class TTEvalArgs(EvalArgs):
    max_input_length: Union[None, int] = None
    max_output_length: Union[None, int] = None

@dataclass
class TTTestArgs(TestArgs):
    pass
