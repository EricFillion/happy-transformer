from happytransformer.to_delete.happy_bert import HappyBERT

happy = HappyBERT()

def test_prediction_options():
    '''
    asserts that trimmed options are sorted by
    likelihood and not order in list
    '''
    predictions = happy.predict_mask(
        'I want crackers and [MASK]',
        options=['death', 'cheese'],
        num_results=1
    )
    print(predictions)
    # top prediction should be cheese and not death
    assert predictions[0]['word'] == 'cheese'