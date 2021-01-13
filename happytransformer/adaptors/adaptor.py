from typing import Type
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase

class Adaptor:
    '''
    Holds a few functions for implementation details.
    Does NOT store any state.
    '''
    @property
    def Tokenizer(self) -> Type[PreTrainedTokenizerBase]:
        raise NotImplementedError()

    @property
    def MaskedLM(self) -> Type[PreTrainedModel]:
        raise NotImplementedError()

    @property
    def NextSentencePrediction(self) -> Type[PreTrainedModel]:
        raise NotImplementedError()

    @property
    def QuestionAnswering(self) -> Type[PreTrainedModel]:
        raise NotImplementedError()

    @property
    def SequenceClassification(self) -> Type[PreTrainedModel]:
        raise NotImplementedError()

    def preprocess_mask_text(self, text: str)-> str:
        return text

    def postprocess_mask_prediction_token(self, text: str) -> str:
        return text