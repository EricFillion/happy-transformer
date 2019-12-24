"""
HappyBERT
"""

import logging

# disable pylint TODO warning
# pylint: disable=W0511
# FineTuning Parts
from happy_transformer.bert_utils import train, switch_to_new, load_and_cache_examples, evaluate
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
    def fine_tune(train_path, test_path):
        logger = logging.getLogger(__name__)
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

        model.resize_token_embeddings(len(tokenizer))
        model.cuda()
        train_dataset = load_and_cache_examples(tokenizer, file_path=train_path)
        logger.info("Training Started")
        global_step, tr_loss = train(train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # Start Eval
        logger.info("Eval Started")
        model, tokenizer = switch_to_new('model')
        model.cuda()
        test_dataset = load_and_cache_examples(tokenizer, file_path=test_path)
        return evaluate(model, tokenizer, test_dataset)
