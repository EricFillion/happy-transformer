from .adaptor import Adaptor
from .berts import RobertaAdaptor, AlbertAdaptor

ADAPTORS = {
    'ALBERT': AlbertAdaptor(),
    'ROBERTA': RobertaAdaptor(),

}

def get_adaptor(model_type:str)->Adaptor:
    if model_type in ADAPTORS:
        return ADAPTORS[model_type]
    else:
        # Default for models with no special cases
        return Adaptor()
