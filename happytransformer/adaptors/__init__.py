from .adaptor import Adaptor
from .berts import BertAdaptor, DistilBertAdaptor, RobertaAdaptor, AlbertAdaptor

ADAPTORS = {
    'BERT': BertAdaptor(),
    'DISTILBERT': DistilBertAdaptor(),
    'ROBERTA': RobertaAdaptor(),
    'ALBERT': AlbertAdaptor()
}

def get_adaptor(model_type:str)->Adaptor:
    if model_type in ADAPTORS:
        return ADAPTORS[model_type]
    else:
        raise ValueError(f'Model type <{model_type}> not currently supported')