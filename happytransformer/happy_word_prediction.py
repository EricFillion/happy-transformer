from typing import List, Optional
from dataclasses import dataclass

from transformers import FillMaskPipeline, AutoModelForMaskedLM, PretrainedConfig, DataCollatorForLanguageModeling
from happytransformer.fine_tuning_util import preprocess_concatenate, EvalResult
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.args import WPTrainArgs, WPEvalArgs
from happytransformer.adaptors import get_adaptor
from typing import Union

@dataclass
class WordPredictionResult:
    token: str
    score: float

class HappyWordPrediction(HappyTransformer):
    """
    A user facing class for text classification
    """
    def __init__(
            self, model_type: str = "DISTILBERT", model_name: str = "distilbert-base-uncased",
            load_path: str ="", use_auth_token: str = None):

        self.adaptor = get_adaptor(model_type)
        model_class = AutoModelForMaskedLM


        super().__init__(model_type, model_name, model_class, load_path=load_path, use_auth_token=use_auth_token)


        self._pipeline = FillMaskPipeline(model=self.model, tokenizer=self.tokenizer, device=self.device)



        self._data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=0.1  # todo modify
                                                        )
        self._t_data_file_type = "text"
        self._type = "wp"

    def predict_mask(self, text: str, targets: Optional[List[str]] = None, top_k: int = 1) -> List[WordPredictionResult]:
        """
        Predict [MASK] tokens in a string.
        targets limit possible guesses if supplied.
        top_k describes number of targets to return*
        *top_k does not apply if targets is supplied
        """
        if not isinstance(text, str):
            raise ValueError('the "text" argument must be a single string')

        text_for_pipeline = self.adaptor.preprocess_mask_text(text)
        answers = self._pipeline(
            text_for_pipeline, 
            targets=targets, top_k=top_k
        )

        fix_token = self.adaptor.postprocess_mask_prediction_token
        return [
            WordPredictionResult(
                token=fix_token(answer["token_str"]), 
                score=answer["score"]
            )
            for answer in answers
        ]

    def train(self, input_filepath: str, eval_filepath: str = "", args: WPTrainArgs =WPTrainArgs()):
        super(HappyWordPrediction, self).train(input_filepath, args, eval_filepath)

    def eval(self, input_filepath, args: WPEvalArgs = WPEvalArgs()) -> EvalResult:
        return super(HappyWordPrediction, self).eval(input_filepath, args)


    def test(self, input_filepath, args=None):
        raise NotImplementedError("test() is currently not available")

    def _tok_function(self, raw_dataset, args: Union[WPTrainArgs, WPEvalArgs]):
        if not args.line_by_line:
            return preprocess_concatenate(tokenizer=self.tokenizer, dataset=raw_dataset,
                                      preprocessing_processes=args.preprocessing_processes, mlm=True)
        else:
            def tokenize_function(example):
                return self.tokenizer(example["text"],
                                 add_special_tokens=True, truncation=True)

            tokenized_dataset = raw_dataset.map(tokenize_function, batched=True,
                                            num_proc=args.preprocessing_processes,
                                            remove_columns=["text"])
            return tokenized_dataset
