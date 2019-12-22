# pylint: disable=W0511
from transformers import XLNetLMHeadModel, XLNetTokenizer

from happy_transformer.happy_transformer import HappyTransformer
from happy_transformer.sequence_classification import SequenceClassifier
from happy_transformer.classifier_utils import classifier_args


class HappyXLNET(HappyTransformer):
    """
    Implementation of XLNET for masked word prediction
    """

    def __init__(self, model='xlnet-large-cased', initial_transformers=[]):
        super().__init__(model, initial_transformers)
        self.mlm = None
        self.seq = None
        self.tokenizer = XLNetTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.model = 'XLNET'

    def _get_masked_language_model(self):
        """
        Initializes the XLNetLMHeadModel transformer
        """
        self.mlm = XLNetLMHeadModel.from_pretrained(self.model_to_use)
        self.mlm.eval()


    def _init_sequence_classifier(self, classifier_name: str):
        args = classifier_args.copy()
        args['model_name'] = classifier_name
        args['model_name'] = "xlnet-base-cased"

        self.seq = SequenceClassifier(args)