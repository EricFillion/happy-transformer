"""
Fine-tune Text-to-text models

Based on the following sources:
1. https://github.com/huggingface/notebooks/blob/master/examples/summarization.ipynb
2. https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py
3. https://github.com/huggingface/transformers/blob/master/examples/pytorch/translation/run_translation.py
4. https://huggingface.co/transformers/
5. https://huggingface.co/docs/datasets/
"""

from dataclasses import dataclass
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import tempfile


@dataclass
class TTTrainArgs:
    """
    Used to adjust the settings when calling HappyTextToText.train()
    """
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    weight_decay: float = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data: bool = False
    load_preprocessed_data_path: str = ""
    preprocessing_processes: int = 1
    batch_size: int = 1
    max_input_length: int = 1024
    max_output_length: int = 1024


@dataclass
class TTEvalArgs:
    """
    Used to adjust the settings when calling HappyTextToText.eval()
    """
    batch_size: int = 1
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data: bool = ""
    load_preprocessed_data_path: str = ""
    preprocessing_processes: int = 1
    max_input_length: int = 1024
    max_output_length: int = 1024


@dataclass
class TTTestArgs:
    """
    Used to adjust the settings when calling HappyTextToText.test()
    """
    save_preprocessed_data: bool = False
    save_preprocessed_data_path: str = ""
    load_preprocessed_data: bool = False
    load_preprocessed_data_path: str = ""

class TTTrainer(HappyTrainer):
    """
    Trainer class for HappyTextToText
    """
    def __init__(self, model, model_type, tokenizer, device, logger):
        super().__init__(model, model_type, tokenizer, device, logger)
        self.__max_input_length = 1024
        self.__max_output_length = 1024


    def train(self, input_filepath, dataclass_args=TTTrainArgs):
        """
        :param input_filepath: A file path to a csv file that contains two columns: text_1 and text_2
        :param dataclass_args: A TTTrainArgs() object
        :return: None
        """

        if dataclass_args.save_preprocessed_data:
            self.logger.info("Saving preprocessed data is currently "
                             "not available for text-to-text models. "
                             )
        if dataclass_args.load_preprocessed_data:
            self.logger.info("Loading preprocessed data is currently "
                             "not available for text-to-text models. "
                             )

        self.logger.info("Preprocessing training data...")

        dataset = load_dataset("csv", data_files={"train": input_filepath}, delimiter=",")

        self.__max_input_length = dataclass_args.max_input_length
        self.__max_output_length = dataclass_args.max_output_length

        tokenized_dataset = dataset.map(
            self.__preprocess_function,
            batched=True,
            num_proc=dataclass_args.preprocessing_processes,
            remove_columns=["input", "target"],
        )

        self.logger.info("Training...")

        # Pads inputs and labels to max length
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # A temp dir is used so any files that are generated are deleted after training
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            training_args = Seq2SeqTrainingArguments(
                tmp_dir_name,
                do_train=True,
                do_eval=False,
                learning_rate=dataclass_args.learning_rate,
                weight_decay=dataclass_args.weight_decay,
                adam_beta1=dataclass_args.adam_beta1,
                adam_beta2=dataclass_args.adam_beta2,
                adam_epsilon=dataclass_args.adam_epsilon,
                max_grad_norm=dataclass_args.max_grad_norm,
                num_train_epochs=dataclass_args.num_train_epochs,
                report_to=["none"],
                save_strategy="no", # no checkpoints are saved
                per_device_train_batch_size=dataclass_args.batch_size
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            trainer.train()


    def eval(self, input_filepath, dataclass_args=TTEvalArgs):
        """
        Evaluates the model by determining the loss.
        :param input_filepath: A file path to a csv file that contains two columns: text_1 and text_2
        :param dataclass_args: A TTEvalArgs() object
        :return: a EvalResult object that contains the loss
        """
        self.logger.info("Preprocessing evaluating data...")
        dataset = load_dataset("csv", data_files={"eval": input_filepath}, delimiter=",")

        self.__max_input_length = dataclass_args.max_input_length
        self.__max_output_length = dataclass_args.max_output_length

        tokenized_dataset = dataset.map(
            self.__preprocess_function,
            batched=True,
            remove_columns=["input", "target"],
            num_proc=dataclass_args.preprocessing_processes,
        )

        # Pads inputs and labels to max length
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # A temp dir is used so any files that are generated are deleted after evaluating
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eval_args = Seq2SeqTrainingArguments(
                tmp_dir_name,
                do_train=False,
                do_eval=True,
                seed=42,
                report_to=["none"],
                per_device_eval_batch_size=dataclass_args.batch_size,
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=eval_args,
                eval_dataset=tokenized_dataset['eval'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            result = trainer.evaluate()
            return EvalResult(loss=result["eval_loss"])

    def __preprocess_function(self, examples):
        """
        :param examples:
        :return:
        """
        model_inputs = self.tokenizer(examples["input"], max_length=self.__max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["target"], max_length=self.__max_output_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs



    def test(self, input_filepath, solve, args=TTTestArgs):
        raise NotImplementedError()
