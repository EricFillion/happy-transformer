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

def test_multi_mask_options():
    MASKS_OPTIONS = [
        ['I','You'],
        ['big','small'],
        ['him','her']
    ]
    options_set = set(
        option
        for mask in MASKS_OPTIONS
        for option in mask
    )
    all_predictions = happy.predict_masks(
        "[MASK] have a [MASK] dog and I love [MASK] so much",
        masks_options=MASKS_OPTIONS
    )
    print(all_predictions)
    assert len(all_predictions) == 3
    assert all(
        prediction.text in options_set
        for mask_predictions in all_predictions
        for prediction in mask_predictions
    )