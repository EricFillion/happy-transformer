from happytransformer import HappyQuestionAnswering
def example_3_0():

    happy_qa_distilbert = HappyQuestionAnswering()  # default
    happy_qa_albert = HappyQuestionAnswering("ALBERT", "mfeb/albert-xxlarge-v2-squad2")
    # good model when using with limited hardware
    happy_qa_bert = HappyQuestionAnswering("BERT", "mrm8488/bert-tiny-5-finetuned-squadv2")


def example_3_1():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.answer_question("Today's date is January 10th, 2021", "What is the date?")
    print(type(result))  # <class 'list'>
    print(result)  # [QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34)]
    print(type(result[0]))  # <class 'happytransformer.happy_question_answering.QuestionAnsweringResult'>
    print(result[0])  # QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34)
    print(result[0].answer)  # January 10th, 2021


def example_3_2():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.answer_question("Today's date is January 10th, 2021", "What is the date?", top_k=2)
    print(type(result))  # <class 'list'>
    print(result)  # [QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34), QuestionAnsweringResult(answer='January 10th', score=0.017306014895439148, start=16, end=28)]
    print(result[1].answer)  # January 10th

def example_3_3():
    happy_qa = HappyQuestionAnswering()
    happy_qa.train("../../data/qa/train-eval.csv")

def example_3_4():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.eval("../../data/qa/train-eval.csv")
    print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
    print(result)  # EvalResult(eval_loss=0.11738169193267822)
    print(result.eval_loss)  # 0.1173816919326782

def example_3_5():
    happy_qa = HappyQuestionAnswering()
    result = happy_qa.test("../../data/qa/test.csv")
    print(type(result))
    print(result)  # [QuestionAnsweringResult(answer='October 31st', score=0.9939756989479065, start=0, end=12), QuestionAnsweringResult(answer='November 23rd', score=0.967872679233551, start=12, end=25)]
    print(result[0])  # QuestionAnsweringResult(answer='October 31st', score=0.9939756989479065, start=0, end=12)
    print(result[0].answer)  # October 31st

def example_3_6():
    happy_qa = HappyQuestionAnswering()
    before_loss = happy_qa.eval("../../data/qa/train-eval.csv").eval_loss
    happy_qa.train("../../data/qa/train-eval.csv")
    after_loss = happy_qa.eval("../../data/qa/train-eval.csv").eval_loss
    print("Before loss: ", before_loss)  # 0.11738169193267822
    print("After loss: ", after_loss)  # 0.00037909045931883156
    # Since after_loss < before_loss, the model learned!
    # Note: typically you evaluate with a separate dataset
    # but for simplicity the same one was used

def main():
    example_3_1()


if __name__ == "__main__":
    main()
