from happytransformer import HappyBERT

happy = HappyBERT()

def test_multi_mask():
    # should give something like
    # "I have a great dog and I love him so much"
    all_predictions = happy.predict_masks(
        "[MASK] have a [MASK] dog and I love [MASK] so much",
        num_results=2
    )
    assert len(all_predictions) == 3
    assert all(
        len(specific_predictions) == 2
        for specific_predictions in all_predictions
    )
    assert all_predictions[0][0].text == 'i'
    assert all_predictions[0][0].probability > 0.5

    assert all_predictions[2][0].text == 'him'

if __name__=='__main__':
    test_multi_mask()