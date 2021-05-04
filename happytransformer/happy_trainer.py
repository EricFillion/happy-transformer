"""
Parent class for training classes, such as TCTrainer and QATrainer
"""
from dataclasses import dataclass
import tempfile
from transformers import TrainingArguments, Trainer

@dataclass
class EvalResult:
    loss: float

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

    @staticmethod
    def _get_training_args(dataclass_args, output_path):
        """
        :param args: a dataclass of arguments for training
        :param output_path: A string to a temporary directory
        :return: A TrainingArguments object
        """
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
            per_device_train_batch_size=dataclass_args.batch_size

        )


    def _run_train(self, dataset, dataclass_args, data_collator):
        """

        :param dataset: a child of torch.utils.data.Dataset
        :param dataclass_args: a dataclass that contains settings
        :return: None
        """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            training_args = self._get_training_args(dataclass_args, tmp_dir_name)
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
