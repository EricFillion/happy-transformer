import torch
from transformers import AutoModelForNextSentencePrediction
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.adaptors import get_adaptor
from happytransformer.qa import ARGS_QA_TRAIN, ARGS_QA_EVAl, ARGS_QA_TEST

class HappyNextSentence(HappyTransformer):
    """
    A user facing class for next sentence prediction
    """
    def __init__(self, model_type="BERT",
                 model_name="bert-base-uncased", load_path: str = ""):

        self.adaptor = get_adaptor(model_type)
        if load_path != "":
            model = AutoModelForNextSentencePrediction.from_pretrained(load_path)
        else:
            model = AutoModelForNextSentencePrediction.from_pretrained(model_name)

        super().__init__(model_type, model_name, model)
        self._pipeline = None
        self._trainer = None

    def predict_next_sentence(self, sentence_a: str, sentence_b: str) -> float:
        """
        Predict the probability that sentence_b follows sentence_a.
        Higher probabilities indicate more coherent sentence pairs.
        """

        encoded = self.tokenizer(sentence_a, sentence_b, return_tensors='pt')
        with torch.no_grad():
            scores = self.model(encoded['input_ids'], token_type_ids=encoded['token_type_ids']).logits[0]

        probabilities = torch.softmax(scores, dim=0)
        # probability that sentence B follows sentence A
        score = probabilities[0].item()

        if self._device == 'cuda':
            torch.cuda.empty_cache()

        return score

    def train(self, input_filepath, args=ARGS_QA_TRAIN):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath, args=ARGS_QA_EVAl):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath, args=ARGS_QA_TEST):
        raise NotImplementedError("test() is currently not available")
