from happytransformer.qa.trainer import QATrainer

from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    RobertaForMaskedLM,
    RobertaTokenizerFast
)

from happytransformer.happy_transformer import HappyTransformer


class HappyWordPrediction(HappyTransformer):
    def __init__(self, model_type="BERT",
                 model_name="bert-large-uncased-whole-word-masking-finetuned-squad", device=None):
        model = None
        tokenizer = None

        if model_type == "BERT":
            model = BertForMaskedLM.from_pretrained(model_name)
            tokenizer = BertTokenizerFast.from_pretrained(model_name)

        elif model_type == "ROBERTA":
            model = RobertaForMaskedLM.from_pretrained(model_name)
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

        super().__init__(model_type, model_name, model, tokenizer, device)

        self._trainer = QATrainer(model,
                                  model_type, tokenizer, self._device, self.logger)
    def predict_masks(self):
        raise NotImplementedError()

    def predict_mask(self):
        raise NotImplementedError()

    def train(self, input_filepath, args):
        raise NotImplementedError()

    def test(self, input_filepath, output_filepath, args):
        raise NotImplementedError()

    def eval(self, input_filepath, output_filepath, args):
        raise NotImplementedError()
