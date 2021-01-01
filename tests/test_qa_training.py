from happytransformer import HappyBERT


def test_qa_training():

    happy_bert = HappyBERT()
    happy_bert.init_qa()
    start_answers = happy_bert.answers_to_question("What is the date?", "October 31st is the date")
    happy_bert.train_qa("../data/sample-qa-training-eval-data.csv")
    end_answers = happy_bert.answers_to_question("What is the date?", "October 31st is the date")

    assert start_answers[0]["text"] == "october 31st"
    assert end_answers[0]["text"] == "october 31st"
    assert end_answers[0]["softmax"] > start_answers[0]["softmax"]

def test_qa_eval():
    happy_bert = HappyBERT()
    happy_bert.init_qa()
    before = happy_bert.eval_qa("../data/sample-qa-training-eval-data.csv")
    happy_bert.train_qa("../data/sample-qa-training-eval-data.csv")
    after = happy_bert.eval_qa("../data/sample-qa-training-eval-data.csv")

    #todo assert by making sure the output csv makes sense
    # todo, also, perhaps use a different dataset for training and eval


    assert after >= before
    # todo get a larger dataset
    # however, we do not want to commit a large dataset to the repo,
    # so we'll have to find a way to download it from the web when the code runs

    # also, use separate data for test and eval
    # assert after > before




# def test_qa_test():
#todo

