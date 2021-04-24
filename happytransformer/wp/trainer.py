"""
Fine-tuning for masked word prediction models.

Based on the tutorial found here:
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
"""

from transformers import DataCollatorForLanguageModeling
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from datasets import load_dataset
from happytransformer.fine_tuning_util import preprocess_concatenate

class WPTrainer(HappyTrainer):
    """
    Trainer class for HappyWordPrediction
    """

    def train(self, input_filepath, args):
        dataset = load_dataset("text", data_files={"train": input_filepath})

        if args["line-by-line"]:
            tokenized_dataset = self._preprocess_line_by_line(self.tokenizer, dataset, args)
        else:
            tokenized_dataset = preprocess_concatenate(self.tokenizer, dataset, args, True)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=args['mlm_probability'])

        self._run_train(tokenized_dataset['train'], args, data_collator)


    def eval(self, input_filepath, args):
        dataset = load_dataset("text", data_files={"eval": input_filepath})

        if args["line-by-line"]:
            tokenized_dataset = self._preprocess_line_by_line(self.tokenizer, dataset, args)
        else:
            tokenized_dataset = preprocess_concatenate(self.tokenizer, dataset, args, True)


        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=args['mlm_probability'])
        result = self._run_eval(tokenized_dataset['eval'], data_collator)

        return EvalResult(loss=result["eval_loss"])


    def test(self, input_filepath, solve, args):
        raise NotImplementedError()


    def _preprocess_line_by_line(self, tokenizer, dataset, args):
        """
        :param tokenizer: tokenizer for a transformer model
        :param datasets: a datasets.Dataset object
        :param args: A dictionary that contains settings
        :return:
        """

        def tokenize_function(example):
            return tokenizer(example["text"],
                             add_special_tokens=True, truncation=True,)

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                          num_proc=args["preprocessing_processes"],
                                          remove_columns=["text"])
        return tokenized_dataset
