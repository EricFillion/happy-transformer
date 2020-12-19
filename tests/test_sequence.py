from happytransformer import HappyBERT

happy = HappyBERT()

def test_sequence_runs():
    '''
    run sequence classification end to end
    only checks that no errors are raised
    '''
    happy.init_sequence_classifier()
    happy.train_sequence_classifier('tests/test_sequence.csv')
    happy.eval_sequence_classifier('tests/test_sequence.csv')