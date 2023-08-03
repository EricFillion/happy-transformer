import logging
import os
import tempfile
from typing import Union

from datasets import Dataset, DatasetDict,  load_dataset, load_from_disk
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments
    )

from happytransformer.args import EvalArgs, TrainArgs
from happytransformer.fine_tuning_util import EvalResult, FistStep, ZERO_2_SETTINGS, ZERO_3_SETTINGS

class HappyTransformer():

    def __init__(self, model_type: str, model_name: str, model_class: AutoModel, load_path="", use_auth_token: Union[str, bool] = None, trust_remote_code: bool =False):

        self.logger = self._get_logger()
        self.model_type = model_type
        self.model_name = model_name

        # Sets self.model and self.tokenizer if load_model is True
        if load_path != "":
            self.logger.warning(f"load_path has been deprecated. Provide the load_path to the  model_name parameter instead {self.model_name}. load_path will be removed in a later version. For now, we'll load the model form the load_path provided.  ")
            self.model_name = load_path

        self.config, self.tokenizer, self.model = self._get_model_components(self.model_name, use_auth_token, trust_remote_code, model_class)

        self.device = self.__get_device()

        self.logger.info("Using device: %s", self.device)

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.resize_token_embeddings(len(self.tokenizer))

        # Set within the child classes.
        self._data_collator = None
        self._t_data_file_type = None
        self._type = None
        self._pipeline_class = None

        # Loaded in upon first time calling text generation.
        self._pipeline = None

        self._on_device = False

    ######## Children of
    def _tok_function(self, raw_dataset, args: TrainArgs, format: str) -> Dataset:
        raise NotImplementedError()

    ######## Helper __init__ methods ########

    def _get_model_components(self, model_name_path,  use_auth_token, trust_remote_code, model_class):
        # HappyTextClassification is the only class that overwrites
        # this as we need to specify number of labels.

        config = AutoConfig.from_pretrained(model_name_path, use_auth_token=use_auth_token, trust_remote_code=trust_remote_code)
        model = model_class.from_pretrained(model_name_path, config=config, use_auth_token=use_auth_token, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_name_path, use_auth_token=use_auth_token, trust_remote_code=trust_remote_code)

        return config, tokenizer, model

    def _get_logger(self) :
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.addFilter(logging.Filter('happytransformer'))
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[handler]
        )
        return logger


    def __get_device(self):
        device = None
        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                device = torch.device("mps")

        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        if not device:
            device = torch.device("cpu")

        return device

    def train(self, input_filepath: str ,  args: TrainArgs, eval_filepath: str = "", ):
        if type(args) == dict:
            raise ValueError("Dictionary training arguments are no longer supported as of Happy Transformer version 3.0.0.")

        if args.eval_ratio <= 0 and eval_filepath == "":
            raise ValueError("Please set TrainArgs.eval_ratio to greater than 0  or supply an eval_path")

        train_tok_data, eval_tok_data = self._preprocess_data_train(input_filepath=input_filepath,
                                                              eval_filepath=eval_filepath,
                                                              args=args)

        self._run_train(train_tok_data, eval_tok_data, args,  self._data_collator)



    def eval(self, input_filepath, args):
        if type(args) == dict:
            raise ValueError(
                "Dictionary evaluating arguments are no longer supported as of Happy Transformer version 3.0.0.")


        tokenized_dataset = self._preprocess_data_eval(input_filepath, args)

        result = self._run_eval(tokenized_dataset, self._data_collator, args)

        return EvalResult(loss=result["eval_loss"])

    def test(self, input_filepath, args):
        raise NotImplementedError()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    ######## Data Preprocessing ########
    def _preprocess_data_train(self, input_filepath, eval_filepath, args: TrainArgs):

        if not args.load_path:
            # We are loading raw data
            if eval_filepath == "":

                # eval_filepath was not provided so we use a portion of the training data for evaluating
                file_type = self._check_file_type(input_filepath)
                all_raw_data = load_dataset(file_type, data_files={"train": input_filepath}, split="train")
                # Shuffle data
                all_raw_data = all_raw_data.shuffle(seed=42)
                # Split according to args.eval_ratio
                split_text_data = all_raw_data.train_test_split(test_size=args.eval_ratio)
                self.logger.info("Tokenizing training data...")

                train_tok_data = self._tok_function(split_text_data["train"], args, file_type)
                eval_tok_data = self._tok_function(split_text_data["test"], args, file_type)
            else:
                # Eval path has been provided so we can load the evaluating data directly.
                train_file_type = self._check_file_type(input_filepath)
                eval_file_type = self._check_file_type(input_filepath)

                if train_file_type != eval_file_type:
                    raise ValueError("Train file-type must be the same as the eval file-type")

                raw_data = load_dataset(train_file_type, data_files={"train": input_filepath, "eval": eval_filepath})

                # todo combine logic below with above tokenizing logic
                self.logger.info("Tokenizing training data...")
                train_tok_data = self._tok_function(raw_data["train"], args, train_file_type)
                self.logger.info("Tokenizing eval data...")
                eval_tok_data = self._tok_function(raw_data["eval"], args, train_file_type)
        else:
            if eval_filepath != "":
                self.logger.warning(f"Eval data will be fetched from {args.load_path} and not {eval_filepath}")

            tok_data = load_from_disk(args.load_path)
            train_tok_data = tok_data["train"]
            eval_tok_data = tok_data["eval"]

        if args.save_path:

            combined_tok = DatasetDict({"train": train_tok_data, "eval": eval_tok_data})
            combined_tok.save_to_disk(args.save_path)

        return train_tok_data, eval_tok_data


    def _preprocess_data_eval(self, input_filepath, args: TrainArgs):
        if not args.load_path:
            self.logger.info("Preprocessing dataset...")
            eval_file_type = self._check_file_type(input_filepath)
            datasets = load_dataset(eval_file_type, data_files={"eval": input_filepath})
            tokenized_dataset = self._tok_function(datasets["eval"], args, eval_file_type)

        else:
            self.logger.info("Loading dataset from %s...", args.load_path)
            tokenized_dataset = load_from_disk(args.load_path +"/eval")

        if args.save_path:
            if args.load_path:
                self.logger.warning("Both save_data and load_data are enabled.")

            self.logger.info("Saving evaluating dataset to %s...", args.save_path)
            save_dataset = DatasetDict({"eval": tokenized_dataset})
            save_dataset.save_to_disk(args.save_path)

        return tokenized_dataset



    def _get_training_args(self, args):
        if self.device.type != "cuda":
            if args.fp16:
                ValueError("fp16 is only available when CUDA/ a GPU is being used. ")

        if self._type == "tt":
            arg_class = Seq2SeqTrainingArguments
        else:
            arg_class = TrainingArguments

        deepspeed = self.__get_deepspeed_config(args, True)

        return arg_class(
            deepspeed=deepspeed,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            report_to=["none"] if len(args.report_to) == 0 else args.report_to,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps,
            evaluation_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps,
            logging_strategy="steps" if args.logging_steps > 0 else "no",
            logging_steps = args.logging_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            fp16=args.fp16,
            use_mps_device= True if self.device.type == "mps" else False,
            run_name=args.run_name,
        )


    def _run_train(self, train_dataset, eval_dataset, args, data_collator):
        """
        :param dataset: a child of torch.utils.data.Dataset
        :param args: a dataclass that contains settings
        :return: None
        """

        # if model has not been moved to device and DeepSpeed is not being used
        if not self._on_device and not args.deepspeed:
                self.logger.info(f"Moving model to {self.device}")
                self.model.to(self.device)
                self._on_device = True

        training_args = self._get_training_args(args)

        os.environ["WANDB_PROJECT"] = args.project_name

        if self._type == "wp":
            self._data_collator.mlm_probability = args.mlm_probability

        if self._type == "tt":
            train_class = Seq2SeqTrainer
        else:
            train_class = Trainer

        trainer = train_class(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.add_callback(FistStep())
        trainer.train()

    def _run_eval(self, dataset, data_collator, args):
        if self._type == "wp":
            self._data_collator.mlm_probability = args.mlm_probability

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eval_args = self._get_eval_args(tmp_dir_name, args)
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=dataset,
                data_collator=data_collator
            )
            return trainer.evaluate()

    def _get_eval_args(self, output_path: str, args: EvalArgs) -> TrainingArguments:
        deepspeed = self.__get_deepspeed_config(args, False)

        return TrainingArguments(
            output_dir=output_path,
            seed=42,
            report_to=['none'],
            per_device_eval_batch_size=args.batch_size,
            use_mps_device=True if self.device.type == "mps" else False,
            deepspeed=deepspeed
        )


    def push(self, repo_name, private=True):
        self.logger.info("Pushing model...")
        self.model.push_to_hub(repo_name, private=private)

        self.logger.info("Pushing tokenizer...")
        self.tokenizer.push_to_hub(repo_name, private=private)

        self.logger.info("Pushing config...")
        self.config.push_to_hub(repo_name, private=private)

    def _check_file_type(self, file_path):
        split_path = os.path.splitext(file_path)

        file_extension = split_path[1]

        # maps the suffix of the file-path to the name that Hugging Face's  load_dataset function uses
        ending_map = {".txt": "text", ".csv": "csv"}
        ending = ending_map[file_extension]

        if ending not in self._t_data_file_type:
            ValueError(f"Invalid file type for {file_path}.")

        return ending

    def __get_deepspeed_config(self, args: Union[TrainArgs, EvalArgs], train: bool = True):

        if isinstance(args.deepspeed, str):
            if args.deepspeed == "ZERO-2":
                if not train:
                    raise ValueError("Use ZERO-3 or a path to custom DeepSpeed settings for evaluating")
                deepspeed = ZERO_2_SETTINGS
            elif args.deepspeed == "ZERO-3":
                deepspeed = ZERO_3_SETTINGS
            else:
                deepspeed = args.deepspeed
        else:
            deepspeed = None

        return deepspeed

    def _load_pipeline(self):

        if self._pipeline_class is not None and self._pipeline is None:

            # if model has not been model has not been moved to device yet
            if not self._on_device:
                self.logger.info(f"Moving model to {self.device}")
                self.model.to(self.device)
                self._on_device = True

            self.logger.info(f"Initializing a pipeline")
            self._pipeline = self._pipeline_class(model=self.model, tokenizer=self.tokenizer, device=self.device)

