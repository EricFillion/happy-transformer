from pytorch_transformers import BertForMaskedLM, BertTokenizer

from happy_transformer.happy_transformer import HappyTransformer


class HappyBERT(HappyTransformer):

    def __init__(self, model='bert-large-uncased'):
        super().__init__()
        self.masked_token = '[MASK]'
        self.separate_token = '[SEP]'
        self.classification_token = '[CLS]'
        self.transformer = BertForMaskedLM.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model)

        self.transformer.eval()
