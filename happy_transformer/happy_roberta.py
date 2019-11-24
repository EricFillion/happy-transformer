
"""
A wrapper over PyTorch's fairseq implementation of RoBERTa
"""

# disable pylint TODO warning
# pylint: disable=W0511

from happy_transformer.happy_transformer import HappyTransformer

from fairseq.models.roberta import RobertaModel

class HappyRoBERTa(HappyTransformer):
    """
    Implementation of RoBERTa for masked word prediction

    """

    def __init__(self, model='roberta.large'):

        # using fairseq
        self.transformer = RobertaModel.from_pretrained(model)
        self.transformer.eval()

        # get WSC model
        # self.transformer_wsc = RobertaModel.from_pretrained('roberta.large.wsc') # TODO fix
        # self.transformer_wsc.eval()

        self.masked_token = "<mask>"


    def predict_mask(self, text: str):
        """
        :param text: a string that contains "<mask>"
        :return: the most likely word for the token and its score
        """

        if not self._text_verification(text):
            return
        mask_index = text.find(self.masked_token)

        predictions = self.transformer.fill_mask(text, topk=1)
        target_prediction = predictions[0]

        score = target_prediction[1]
        filled_in_sentence = target_prediction[0]
        start_prediction = filled_in_sentence[mask_index:]

        prediction = start_prediction.partition(' ')[0]

        results = [prediction, score]

        return results


    def wsc(self, text):
        # todo load in pretrained wsc model so this can be used
        # print(self.transformer_wsc.disambiguate_pronoun(text))
        return ''
