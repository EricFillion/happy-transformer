from happytransformer import HappyBERT, HappyROBERTA, HappyXLNET

MLM_TRANSFORMERS = [
    HappyBERT,
    HappyROBERTA,
    # HappyXLNET # performance is not great, omitting
]

def _test_mlm_model(transformer_class):
    transformer = transformer_class()
    prediction = transformer.predict_mask('[MASK] have a dog')
    assert prediction[0]['word'].lower() == 'i'

def test_all_mlm_models():
    for transformer_class in MLM_TRANSFORMERS:
        print(f'Testing class {transformer_class.__name__}')
        _test_mlm_model(transformer_class)
