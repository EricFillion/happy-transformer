from typing import List,Optional
from dataclasses import dataclass

from transformers import FillMaskPipeline

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.mwp.trainer import WPTrainer
from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.adaptors import get_adaptor

@dataclass
class WordPredictionResult:
    token: str
    score: float

class HappyWordPrediction(HappyTransformer):
    """
    A user facing class for text classification
    """
    def __init__(
        self, model_type: str = "DISTILBERT", model_name: str = "distilbert-base-uncased"):

        self.adaptor = get_adaptor(model_type)
        model = self.adaptor.MaskedLM.from_pretrained(model_name)
        tokenizer = self.adaptor.Tokenizer.from_pretrained(model_name)

        super().__init__(model_type, model_name, model, tokenizer)

        device_number = detect_cuda_device_number()

        self._pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer, device=device_number)

        self._trainer = WPTrainer(model, model_type, tokenizer, self._device, self.logger)

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

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
