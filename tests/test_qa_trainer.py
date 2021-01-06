"""
Tests for the question answering training, evaluating and testing functionality
"""

from happytransformer.happy_question_answering import HappyQuestionAnswering


def test_qa_train_eval():
    """
    Tests
    HappyQuestionAnswering.eval()
    HappyQuestionAnswering.train()

    """
    happy_qa = HappyQuestionAnswering()
    # Test 1
    start_answers = happy_qa.answers_to_question("What is the date?", "October 31st is the date")

    # Test 2
    before = happy_qa.eval("../data/test_qa_trainer/sample-qa-training-eval-data.csv",
                           output_filepath="../data/test_qa_trainer/results/output-eval-before-qa.csv")

    happy_qa.train("../data/test_qa_trainer/sample-qa-training-eval-data.csv")


    # Test 1
    end_answers = happy_qa.answers_to_question("What is the date?", "October 31st is the date")
    assert start_answers[0]["text"] == "october 31st"
    assert end_answers[0]["text"] == "october 31st"
    assert end_answers[0]["softmax"] > start_answers[0]["softmax"]

    # Test 2
    after = happy_qa.eval("../data/test_qa_trainer/sample-qa-training-eval-data.csv",
                          output_filepath="../data/test_qa_trainer/results/output-eval-after-qa.csv")
    assert after >= before

    # Test 3:
    #todo ensure the output csv file makes sense for eval


def test_qa_testing():
    """
    tests:

    HappyQuestionAnswering.test()

    """

    happy_qa = HappyQuestionAnswering()
    happy_qa.test("../data/test_qa_trainer/sample-qa-test-data.csv",
                  output_filepath="../data/test_qa_trainer/results/output-test-output.csv")

    #todo ensure the output csv file makes sense


