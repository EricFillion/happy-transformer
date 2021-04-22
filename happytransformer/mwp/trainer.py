"""
Fine-tuning for masked word prediction models.

Based on the tutorial found here:
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
"""

from transformers import  DataCollatorForLanguageModeling
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from happytransformer.fine_tuning_util import preprocess_lm_data
from datasets import load_dataset


class WPTrainer(HappyTrainer):
    """
    Trainer class for HappyWordPrediction
    """

    def train(self, input_filepath, args):
        datasets = load_dataset("text", data_files={"train": input_filepath})
        tokenized_datasets = preprocess_lm_data(self.tokenizer, datasets, args)

        print("training...")
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=args['mlm_probability'])

        self._run_train(tokenized_datasets['train'], args, data_collator)


    def eval(self, input_filepath, args):
        datasets = load_dataset("text", data_files={"eval": input_filepath})
        tokenized_datasets = preprocess_lm_data(self.tokenizer, datasets, args)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=args['mlm_probability'])
        result = self._run_eval(tokenized_datasets['eval'], data_collator)

        return EvalResult(loss=result["eval_loss"])


    def test(self, input_filepath, pipeline):
        raise NotImplementedError()
