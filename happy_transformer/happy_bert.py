"""
HappyBERT
"""

# disable pylint TODO warning
# pylint: disable=W0511
# FineTuning Parts
from happy_transformer.happy_transformer import HappyTransformer
from transformers import BertForMaskedLM, BertForNextSentencePrediction, BertTokenizer


class HappyBERT(HappyTransformer):
    """
    A wrapper over PyTorch's BERT transformer implementation
    """

    def __init__(self, model='bert-large-uncased', initial_transformers=[]):
        super().__init__(model, initial_transformers)
        self.model = None  # Masked Language Model
        self.nsp = None  # Next Sentence Prediction
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.model = 'BERT'

    def _get_masked_language_model(self):
        """
        Initializes the BertForMaskedLM transformer
        """
        self.model = BertForMaskedLM.from_pretrained(self.model_to_use)
        self.model.eval()

    def _get_next_sentence_prediction(self):
        """
        Initializes the BertForNextSentencePrediction transformer
        """
        self.nsp = BertForNextSentencePrediction.from_pretrained(self.model_to_use)
        self.nsp.eval()

    def _get_prediction_softmax(self, text: str):
        """
        BERT's "_get_prediction_softmax" is the default in HappyTransformer
        """
        return super()._get_prediction_softmax(text)

    @staticmethod
    def fine_tune(train_path, test_path, batch_size=1, epochs=1, lr=5e-5, adam_epsilon=1e-8):
        from happy_transformer.bert_utils import (train, switch_to_new, create_dataset, evaluate)
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model.resize_token_embeddings(len(tokenizer))  # To make sure embedding size agrees with the tokenizer

        # Start Train
        model.cuda()
        train_dataset = create_dataset(tokenizer, file_path=train_path)
        train(train_dataset, model, tokenizer, batch_size=batch_size, epochs=epochs, lr=lr, adam_epsilon=adam_epsilon)

        # Start Eval
        model, tokenizer = switch_to_new('model')
        model.cuda()
        test_dataset = create_dataset(tokenizer, file_path=test_path, batch_size=2)
        return evaluate(model, tokenizer, test_dataset)
