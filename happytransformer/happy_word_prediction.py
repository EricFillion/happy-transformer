from typing import List, Optional
from dataclasses import dataclass

from transformers import FillMaskPipeline, AutoModelForMaskedLM, PretrainedConfig

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.wp.trainer import WPTrainer, WPTrainArgs, WPEvalArgs
from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.adaptors import get_adaptor
from happytransformer.wp import ARGS_WP_TRAIN, ARGS_WP_EVAl, ARGS_WP_TEST
from happytransformer.happy_trainer import EvalResult
from happytransformer.fine_tuning_util import create_args_dataclass

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
            load_path: str =""):


        self.adaptor = get_adaptor(model_type)

        if load_path != "":
            model = AutoModelForMaskedLM.from_pretrained(load_path)
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_name)

        super().__init__(model_type, model_name, model, load_path=load_path)

        device_number = detect_cuda_device_number()

        self._pipeline = FillMaskPipeline(model=self.model, tokenizer=self.tokenizer, device=device_number)

        self._trainer = WPTrainer(self.model, model_type, self.tokenizer, self._device, self.logger)

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

    def train(self, input_filepath, args=ARGS_WP_TRAIN):
        if type(args) == dict:
            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_WP_TRAIN,
                                                         input_dic_args=args,
                                                         method_dataclass_args=WPTrainArgs)
        elif type(args) == WPTrainArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a WPTrainArgs object or a dictionary")

        self._trainer.train(input_filepath=input_filepath, dataclass_args=method_dataclass_args)

    def eval(self, input_filepath, args=ARGS_WP_EVAl) -> EvalResult:
        if type(args) == dict:

            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_WP_EVAl,
                                                         input_dic_args=args,
                                                         method_dataclass_args=WPEvalArgs)
        elif type(args) == WPEvalArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a ARGS_WP_EVAl object or a dictionary")

        return self._trainer.eval(input_filepath=input_filepath, dataclass_args=method_dataclass_args)


    def test(self, input_filepath, args=ARGS_WP_TEST):
        raise NotImplementedError("test() is currently not available")
