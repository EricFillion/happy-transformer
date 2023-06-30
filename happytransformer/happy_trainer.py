"""
Parent class for training classes, such as TCTrainer and QATrainer
"""
import torch
from dataclasses import dataclass
import tempfile
from transformers import TrainingArguments, Trainer
import math

@dataclass
class EvalResult:
    loss: float

"""
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
 """

@dataclass
class TrainArgs:
    learning_rate: float = 5e-5
    num_train_epochs: int = 3.0
    batch_size: int = 1
    gas: int = 1
    weight_decay: float = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm:  float = 1.0
    fp16: bool = False
    eval_per_epoch: int = 2



class HappyTrainer:
    def __init__(self, model, model_type, tokenizer, device, logger):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger

    def train(self, input_filepath, args):
        """

        :param input_filepath: A string to file location
        :param args: a dictionary that contains settings
        :return:
        """
        raise NotImplementedError()

    def test(self, input_filepath, solve, args):
        """

        :param input_filepath: A string to file location
        :param solve: a method for using the model for the given task
        :return: test results
        """
        raise NotImplementedError()

    def eval(self, input_filepath, args):
        """
        :param input_filepath: A string to file location
        :args a dictionary that contains settings
        :return: a dictionary that contains a key called "eval_loss" that holds the loss
         for the given eval dataset. May add more metrics later
        """
        raise NotImplementedError()

    @staticmethod
    def _get_data(filepath, test_data=False):
        """

        :param filepath:  A string to file location
        :param test_data: False for train and eval, True for test
        :return: varies for each task
        """
        raise NotImplementedError()

    def _get_training_args(self, dataclass_args, output_path, data_len ):
        """
        :param args: a dataclass of arguments for training
        :param output_path: A string to a temporary directory
        :return: A TrainingArguments object
        """
        if self.device != "cuda":
            if dataclass_args.fp16:
                ValueError("fp16 is only available when CUDA/ a GPU is being used. ")

        eval_steps = action_step(
            ape=dataclass_args.eval_per_epoch,
            batch_size=dataclass_args.batch_size,
            gas=dataclass_args.gas,
            data_len=data_len,
            num_gpus= 1 # todo make this adjustable
        )

        return TrainingArguments(
            output_dir=output_path,
            learning_rate=dataclass_args.learning_rate,
            weight_decay=dataclass_args.weight_decay,
            adam_beta1=dataclass_args.adam_beta1,
            adam_beta2=dataclass_args.adam_beta2,
            adam_epsilon=dataclass_args.adam_epsilon,
            max_grad_norm=dataclass_args.max_grad_norm,
            num_train_epochs=dataclass_args.num_train_epochs,
            report_to=["none"],
            save_strategy="no",
            # todo enable after supporting eval dataset
            #evaluation_strategy="steps",
            #eval_steps=eval_steps,
            per_device_train_batch_size=dataclass_args.batch_size,
            fp16=dataclass_args.fp16,
            gradient_accumulation_steps=dataclass_args.gas
        )


    def _run_train(self, dataset, dataclass_args, data_collator):
        """

        :param dataset: a child of torch.utils.data.Dataset
        :param dataclass_args: a dataclass that contains settings
        :return: None
        """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            training_args = self._get_training_args(dataclass_args, tmp_dir_name, len(dataset))
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            trainer.train()
    def _run_eval(self, dataset, data_collator, dataclass_args):
        """
        :param dataset: a child of torch.utils.data.Dataset
        :return: None
        """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eval_args = self._get_eval_args(tmp_dir_name, dataclass_args)
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=dataset,
                data_collator=data_collator
            )
            return trainer.evaluate()

    @staticmethod
    def _get_eval_args(output_path, dataclass_args):
        """
        :param output_path: A string to a temporary directory
        :return: A TrainingArguments object
        """
        return TrainingArguments(
            output_dir=output_path,
            seed=42,
            report_to=['none'],
            per_device_eval_batch_size=dataclass_args.batch_size,

        )


def action_step(ape, batch_size, gas, data_len, num_gpus) -> int:
    """
    :param ape: The number of actions per epoch (save, eval or log).
    :param batch_size: The batch size.
    :param gas: Gradient accumulation steps
    :param data_len: Number of cases within the  training data
    :param num_gpus: Number of GPUs
    :return: The numbner
    """
    epoch_step_len = data_len / (batch_size * gas * num_gpus)

    action_step = math.ceil(epoch_step_len / ape)

    return action_step