from happytransformer import HappyBERT, HappyROBERTA, HappyXLNET

MLM_TRANSFORMERS = [
    HappyBERT,
    HappyROBERTA,
    HappyXLNET
]

def _test_mlm_model(transformer_class):
    transformer = transformer_class()
    prediction = transformer.predict_mask('[MASK] have a dog')
    assert prediction['text'].lower() == 'i'

def test_all_mlm_models():
    for transformer_class in MLM_TRANSFORMERS:
        _test_mlm_model(transformer_class)