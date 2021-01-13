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
        # this should be NotImplementedError because
        # all Adaptors should have a Tokenizer
        raise NotImplementedError()

    @property
    def MaskedLM(self) -> Type[PreTrainedModel]:
        raise ValueError('This model does not support word prediction')

    @property
    def NextSentencePrediction(self) -> Type[PreTrainedModel]:
        raise ValueError('This model does not support next sentence prediction')

    @property
    def QuestionAnswering(self) -> Type[PreTrainedModel]:
        raise ValueError('This model does not support question answering')

    @property
    def SequenceClassification(self) -> Type[PreTrainedModel]:
        raise ValueError('This model does not support sequence classification')

    def preprocess_mask_text(self, text: str)-> str:
        return text

    def postprocess_mask_prediction_token(self, text: str) -> str:
        return text