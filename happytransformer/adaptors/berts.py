from .adaptor import Adaptor



class RobertaAdaptor(Adaptor):

    @staticmethod
    def preprocess_mask_text(text: str) -> str:
        return text.replace('[MASK]', '<mask>')

    @staticmethod
    def postprocess_mask_prediction_token(text):
        return text[1:] if text[0] == "Ġ" else text

class AlbertAdaptor(Adaptor):
    @staticmethod
    def postprocess_mask_prediction_token(text):
        return text[1:] if text[0] == "▁" else text