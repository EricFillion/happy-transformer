
class Adaptor:
    '''
    Holds a few functions for implementation details.
    Does NOT store any state.
    '''

    @staticmethod
    def preprocess_mask_text(text: str) -> str:
        return text

    @staticmethod
    def postprocess_mask_prediction_token(text: str) -> str:
        return text