"""
Parent class for training classes, such as TCTrainer and QATrainer
"""

import time
import datetime
import math
import tempfile
from csv import DictWriter
from transformers import TrainingArguments, Trainer


class HappyTrainer:
    def __init__(self, model, model_type, tokenizer, device, logger):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger


    def train(self, input_filepath, args):
        raise NotImplementedError()

    def test(self, input_filepath):
        raise NotImplementedError()

    def eval(self, input_filepath):
        raise NotImplementedError()

    @staticmethod
    def _get_data(filepath, test_data=False):
        raise NotImplementedError()

    @staticmethod
    def _get_training_args(args, output_path):
        """
        :param args: a dictionary of arguments for training
        :param output_path: A string to a temporary directory
        :return: A TrainingArguments object
        """
        return TrainingArguments(
            output_dir=output_path,
            learning_rate=args["learning_rate"],
            weight_decay=args["weight_decay"],
            adam_beta1=args["adam_beta1"],
            adam_beta2=args["adam_beta2"],
            adam_epsilon=args["adam_epsilon"],
            max_grad_norm=args["max_grad_norm"],
            num_train_epochs=args["num_train_epochs"],

        )

    def _run_train(self, dataset, args):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            training_args = self._get_training_args(args, tmp_dir_name)
            trainer = Trainer(
                model=self.model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=dataset,  # training dataset
            )
            trainer.train()

    def _run_eval(self, dataset):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eval_args = self._get_test_eval_args(tmp_dir_name)

            trainer = Trainer(
                model=self.model,  # the instantiated 🤗 Transformers model to be trained
                args=eval_args,
                eval_dataset=dataset,  # training dataset

            )

            return trainer.evaluate()



    @staticmethod
    def _get_test_eval_args(output_path):
        """

        :param output_path: A string to a temporary directory
        :return: A TrainingArguments object
        """
        return TrainingArguments(
            output_dir=output_path,
            seed=42

        )

    def _format_time(self, time):
        """
        elapsed: time in seconds
        return: time outputted in hh:mm:ss format
        """
        time_rounded = int(round((time)))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=time_rounded))


    def _get_update_interval(self, count):
        """
        Determines how often to print status, given the number of cases.

        First determines how often to update for exactly 50 updates.
        Then, rounds to the nearest power of ten (10, 100, 1000 etc)

        :param count:
        :return:
        """

        x = count / 50
        order = math.floor(math.log(x, 10))

        update_interval = 10 ** order
        if update_interval == 0:
            return 1
        return update_interval

    def _print_status(self, init_time, count, total, update_interval, percentage=None):
        if count % update_interval and not count == 0:
            current_time = time.time()
            elapsed_time_string = self._format_time(current_time - init_time)

            avg_ex = (current_time - init_time) / count
            rem_time_int = avg_ex * (total - count)
            rem_time_string = self._format_time(rem_time_int)
            ending = ""
            if percentage is not None:
                ending = "Correct: " + str(round(percentage, 2)*100) + "%"
            status_output = "Done: ", str(count) + "/" + str(
                total) + "  ----  Elapsed: " + elapsed_time_string +\
                            "   Estimated Remaining: " + rem_time_string +"  " + ending
            self.logger.info(status_output)

    def _output_result_to_csv(self, output_filepath, fieldnames, results):
        with open(output_filepath, 'w') as csv_file:
            csv_writer = DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for result in results:
                csv_writer.writerow(
                    result
                )

