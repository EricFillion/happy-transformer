"""
HappyRoBERTa
"""


# disable pylint TODO warning
# pylint: disable=W0511

from happy_transformer.happy_transformer import HappyTransformer

from transformers import RobertaForMaskedLM, RobertaTokenizer

class HappyRoBERTa(HappyTransformer):
    """
    A wrapper over PyTorch's BERT transformer implementation
    """

    def __init__(self, model='roberta-base', initial_transformers=[]):
        super().__init__(model, initial_transformers)
        self.mlm = None  # Masked Language Model
        self.nsp = None  # Next Sentence Prediction
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.model = 'RoBERTa'

    def _get_masked_language_model(self):
        """
        Initializes the RoBERTaForMaskedLM transformer
        """
        self.mlm = RobertaForMaskedLM.from_pretrained(self.model_to_use)
        self.mlm.eval()
