"""
HappyXLNET: a wrapper over PyTorch's XLNet implementation
"""

from transformers import (
    XLNetLMHeadModel,
    XLNetTokenizer
)

from happytransformer.happy_transformer import HappyTransformer


class HappyXLNET(HappyTransformer):
    """
    Currently available public methods:
        XLNetLMHeadModel:
            1. predict_mask(text: str, options=None, k=1)
        XLNetForSequenceClassification:
            1. init_sequence_classifier()
            2. advanced_init_sequence_classifier()
            3. train_sequence_classifier(train_csv_path)
            4. eval_sequence_classifier(eval_csv_path)
            5. test_sequence_classifier(test_csv_path)

    """

    def __init__(self, model='xlnet-base-cased'):
        super().__init__(model, "XLNET")
        self.mlm = None
        self.tokenizer = XLNetTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token

    def _get_masked_language_model(self):
        """
        Initializes the XLNetLMHeadModel transformer
        """
        self.mlm = XLNetLMHeadModel.from_pretrained(self.model)
        self.mlm.eval()
