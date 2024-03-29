from happytransformer import HappyTokenClassification
from happytransformer.happy_token_classification import TokenClassificationResult
from tests import happy_toc

def test_classify_text():
    expected_result = [TokenClassificationResult(word='Geoffrey', score=0.9988969564437866, entity='B-PER', index=4, start=11, end=19), TokenClassificationResult(word='Toronto', score=0.9993201494216919, entity='B-LOC', index=9, start=34, end=41)]

    result = happy_toc.classify_token("My name is Geoffrey and I live in Toronto")
    assert result == expected_result

def test_toc_save():
    happy_toc.save("model/")
    result_before = happy_toc.classify_token("My name is Geoffrey and I live in Toronto")

    happy = HappyTokenClassification(load_path="model/")
    result_after = happy.classify_token("My name is Geoffrey and I live in Toronto")

    assert result_before[0].word==result_after[0].word
