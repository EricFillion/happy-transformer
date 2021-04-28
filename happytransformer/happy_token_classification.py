from typing import List, Optional
from dataclasses import dataclass

from transformers import TokenClassificationPipeline, AutoModelForTokenClassification

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.toc.trainer import TOCTrainer
from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.adaptors import get_adaptor
from happytransformer.toc import  ARGS_TOC_TRAIN, ARGS_TOC_EVAl, ARGS_TOC_TEST


@dataclass
class TokenClassificationResult:
    word: str
    score: float
    entity: str
    index: int
    start: int
    end: int


class HappyTokenClassification(HappyTransformer):
    """
    A user facing class for text classification
    """
    def __init__(
        self, model_type: str = "BERT", model_name: str = "dslim/bert-base-NER", load_path: str = ""):

        self.adaptor = get_adaptor(model_type)

        if load_path != "":
            model = AutoModelForTokenClassification.from_pretrained(load_path)
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_name)


        super().__init__(model_type, model_name, model)

        device_number = detect_cuda_device_number()

        self._pipeline = TokenClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=device_number)

        self._trainer = TOCTrainer(self.model, model_type, self.tokenizer, self._device, self.logger)

    def classify_token(self, text: str) -> List[TokenClassificationResult]:
        """
        :param text: Text that contains tokens to be classified
        :return:
        """
        if not isinstance(text, str):
            raise ValueError('the "text" argument must be a single string')
        results = self._pipeline(text)

        return [
            TokenClassificationResult(
                word=answer["word"],
                score=answer["score"],
                entity=answer["entity"],
                index=answer["index"],
                start=answer["start"],
                end=answer["end"],
            )
            for answer in results
        ]


    def train(self, input_filepath, args=ARGS_TOC_TRAIN):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath, args=ARGS_TOC_EVAl):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath, args=ARGS_TOC_TEST):
        raise NotImplementedError("test() is currently not available")
