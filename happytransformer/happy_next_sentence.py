import torch

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.adaptors import get_adaptor

class HappyNextSentence(HappyTransformer):
    """
    A user facing class for next sentence prediction
    """
    def __init__(self, model_type="BERT",
                 model_name="bert-base-uncased"):

        self.adaptor = get_adaptor(model_type)
        model = self.adaptor.NextSentencePrediction.from_pretrained(model_name)
        tokenizer = self.adaptor.Tokenizer.from_pretrained(model_name)
        super().__init__(model_type, model_name, model, tokenizer)
        self._pipeline = None
        self._trainer = None

    def predict_next_sentence(self, sentence_a: str, sentence_b: str) -> float:
        """
        Predict the probability that sentence_b follows sentence_a.
        Higher probabilities indicate more coherent sentence pairs.
        """

        encoded = self._tokenizer(sentence_a, sentence_b, return_tensors='pt')
        with torch.no_grad():
            scores = self._model(encoded['input_ids'], token_type_ids=encoded['token_type_ids']).logits[0]

        probabilities = torch.softmax(scores, dim=0)
        # probability that sentence B follows sentence A
        score = probabilities[0].item()

        if self._device == 'cuda':
            torch.cuda.empty_cache()

        return score

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
