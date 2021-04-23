"""

Fine-tuning for text generation odels.

Based on the tutorial found here:
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
"""

from transformers import default_data_collator
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from happytransformer.fine_tuning_util import preprocess_concatenate
from datasets import load_dataset


class GENTrainer(HappyTrainer):
    """
    Trainer class for HappyWordPrediction
    """

    def train(self, input_filepath, args):
        datasets = load_dataset("text", data_files={"train": input_filepath})
        tokenized_datasets = preprocess_concatenate(self.tokenizer, datasets, args, False)

        print("training...")
        self._run_train(tokenized_datasets['train'], args, default_data_collator)


    def eval(self, input_filepath, args):
        datasets = load_dataset("text", data_files={"eval": input_filepath})
        tokenized_datasets = preprocess_concatenate(self.tokenizer, datasets, args, False)

        result = self._run_eval(tokenized_datasets['eval'], default_data_collator)

        return EvalResult(loss=result["eval_loss"])


    def test(self, input_filepath, solve, args):
        raise NotImplementedError()
