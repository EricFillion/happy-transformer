from happytransformer import HappyTextClassification
from happytransformer import HappyTokenClassification


def example_2_0():
    happy_tc_distilbert = HappyTextClassification()  # default with "distilbert-base-uncased"
    happy_tc_albert = HappyTextClassification(model_type="ALBERT", model_name="albert-base-v2")
    happy_tc_bert = HappyTextClassification("BERT", "bert-base-uncased")
    happy_tc_roberta = HappyTextClassification("ROBERTA", "deepset/roberta-base-squad2")


def example_2_1():
    happy_tc = HappyTextClassification(model_type="DISTILBERT",  model_name="distilbert-base-uncased-finetuned-sst-2-english")
    result = happy_tc.classify_text("Great movie! 5/5")
    print(type(result))  # <class 'happytransformer.happy_text_classification.TextClassificationResult'>
    print(result)  # TextClassificationResult(label='LABEL_1', score=0.9998761415481567)
    print(result.label)  # LABEL_1


def example_2_2():
    happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                       model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                       num_labels=2)  # Don't forget to set num_labels!
    happy_tc.train("../../data/tc/train-eval.csv")


def example_2_3():
    happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                       model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                       num_labels=2)  # Don't forget to set num_labels!
    result = happy_tc.eval("../../data/tc/train-eval.csv")
    print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
    print(result)  # EvalResult(eval_loss=0.007262040860950947)
    print(result.loss)  # 0.007262040860950947


def example_2_4():
    happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                       model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                       num_labels=2)  # Don't forget to set num_labels!
    result = happy_tc.test("../../data/tc/test.csv")
    print(type(result))  # <class 'list'>
    print(result)  # [TextClassificationResult(label='LABEL_1', score=0.9998401999473572), TextClassificationResult(label='LABEL_0', score=0.9772131443023682)...
    print(type(result[0]))  # <class 'happytransformer.happy_text_classification.TextClassificationResult'>
    print(result[0])  # TextClassificationResult(label='LABEL_1', score=0.9998401999473572)
    print(result[0].label)  # LABEL_1


def example_2_5():
    happy_tc = HappyTextClassification(model_type="DISTILBERT",
                                       model_name="distilbert-base-uncased-finetuned-sst-2-english",
                                       num_labels=2)  # Don't forget to set num_labels!
    before_loss = happy_tc.eval("../../data/tc/train-eval.csv").loss
    happy_tc.train("../../data/tc/train-eval.csv")
    after_loss = happy_tc.eval("../../data/tc/train-eval.csv").loss
    print("Before loss: ", before_loss)  # 0.007262040860950947
    print("After loss: ", after_loss)  # 0.000162081079906784
    # Since after_loss < before_loss, the model learned!
    # Note: typically you evaluate with a separate dataset
    # but for simplicity we used the same one

def example_5_1():
    happy_toc = HappyTokenClassification(model_type="BERT", model_name="dslim/bert-base-NER")
    result = happy_toc.classify_token("My name is Geoffrey and I live in Toronto")
    print(type(result))  # <class 'list'>
    print(result[0].word)  # Geoffrey
    print(result[0].entity)  # B-PER
    print(result[0].score)  # 0.9988969564437866
    print(result[0].index)  # 4
    print(result[0].start) # 11
    print(result[0].end)  # 19

    print(result[1].word)  # Toronto
    print(result[1].entity)  # B-LOC


def main():
    example_5_1()


if __name__ == "__main__":
    main()
