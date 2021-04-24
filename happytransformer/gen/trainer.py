"""

Fine-tuning for text generation odels.

Based on the tutorial found here:
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
"""
import json
from transformers import default_data_collator
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from happytransformer.fine_tuning_util import preprocess_concatenate
from datasets import load_dataset

class GENTrainer(HappyTrainer):
    """
    Trainer class for HappyWordPrediction
    """

    def train(self, input_filepath, args):

        if not args["load_data"]:
            self.logger.info("Preprocessing dataset...")
            dataset = load_dataset("text", data_files={"train": input_filepath})
            tokenized_dataset = preprocess_concatenate(self.tokenizer, dataset, args, False)

        else:
            self.logger.info("Loading dataset from %s...", args["load_data_path"])
            tokenized_dataset = load_dataset("json", data_files={"train": args["load_data_path"]}, field='train')

        if args['save_data']:
            if args['load_data']:
                self.logger.warning("Both save_data and load_data are enabled,")

            self.logger.info("Saving training dataset to %s...", args["save_data_path"])

            self._generate_json(args['save_data_path'], tokenized_dataset["train"], "train")

        self.logger.info("Training...")

        self._run_train(tokenized_dataset['train'], args, default_data_collator)

    def eval(self, input_filepath, args):

        if not args["load_data"]:
            self.logger.info("Preprocessing dataset...")
            datasets = load_dataset("text", data_files={"eval": input_filepath})
            tokenized_dataset = preprocess_concatenate(self.tokenizer, datasets, args, False)

        else:
            self.logger.info("Loading dataset from %s...", args["load_data_path"])
            tokenized_dataset = load_dataset("json", data_files={"eval": args["load_data_path"]}, field='eval')

        if args['save_data']:
            if args['load_data']:
                self.logger.warning("Both save_data and load_data are enabled.")

            self.logger.info("Saving evaluating dataset to %s...", args["save_data_path"])

            self._generate_json(args['save_data_path'], tokenized_dataset["eval"], "eval")

        self.logger.info("Evaluating...")

        result = self._run_eval(tokenized_dataset['eval'], default_data_collator)

        return EvalResult(loss=result["eval_loss"])

    def test(self, input_filepath, solve, args):
        raise NotImplementedError()

    def _generate_json(self, json_path, dataset, name):
        data = {}
        data[name] = []
        for case in dataset:
            data[name].append({'attention_mask': case["attention_mask"],
                               'input_ids': case["input_ids"],
                               'labels': case["labels"],
                               })
        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)
