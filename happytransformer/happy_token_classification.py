from typing import List, Optional
from dataclasses import dataclass

from transformers import TokenClassificationPipeline, AutoModelForTokenClassification

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.adaptors import get_adaptor
from typing import Union

@dataclass
class TokenClassificationResult:
    word: str
    score: float
    entity: str
    index: int
    start: int
    end: int


class HappyTokenClassification(HappyTransformer):
    def __init__(
        self, model_type: str = "BERT", model_name: str = "dslim/bert-base-NER", load_path: str = "", use_auth_token: Union[bool, str] = None):

        self.adaptor = get_adaptor(model_type)
        model_class = AutoModelForTokenClassification

        super().__init__(model_type, model_name, model_class, use_auth_token=use_auth_token, load_path=load_path)

        self._pipeline = TokenClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=self.device)

        self._type = "tok"

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


    def train(self, input_filepath, eval_filepath, args=None):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath, args=None):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath, args=None):
        raise NotImplementedError("test() is currently not available")
