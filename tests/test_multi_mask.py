from happytransformer import HappyBERT

happy = HappyBERT()

def test_multi_mask():
    all_predictions = happy.predict_masks(
        "[MASK] have a [MASK] dog and I love [MASK] so much"
    )
    print(all_predictions)

if __name__=='__main__':
    test_multi_mask()