"""
Tests for the question answering training, evaluating and testing functionality
"""

from happytransformer.happy_text_classification import HappyTextClassification


def test_qa_train():
    """
    Tests
    HappyQuestionAnswering.eval()
    HappyQuestionAnswering.train()

    """
    happy_tc = HappyTextClassification()
    # Test 1

    # Test 2
    happy_tc.train("../data/test_tc_trainer/sample-tc-training-eval-data.csv", "../data/test_tc_trainer/results/test_qa_train")


def test_qa_eval():
    """
    Tests
    HappyQuestionAnswering.eval()
    HappyQuestionAnswering.train()

    """
    happy_tc = HappyTextClassification()
    results = happy_tc.eval("../data/test_tc_trainer/sample-tc-training-eval-data.csv", "../data/test_tc_trainer/results/test_qa_eval")
    print(results)
    # happy_tc.train("../data/test_tc_trainer/sample-tc-training-eval-data.csv", "../data/test_tc_trainer/results/test_qa_train")
    #results = happy_tc.eval("../data/test_tc_trainer/sample-tc-training-eval-data.csv", "../data/test_tc_trainer/results/test_qa_eval")



def test_qa_test():
    happy_tc = HappyTextClassification()

    happy_tc.test("../data/test_tc_trainer/sample-tc-test-data.csv", "../data/test_tc_trainer/results/test_qa_test")
