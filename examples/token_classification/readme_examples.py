from happytransformer import HappyTokenClassification


def example_5_0():
    happy_toc = HappyTokenClassification("BERT", "dslim/bert-base-NER")  # default
    happy_toc_large = HappyTokenClassification("XLM-ROBERTA", "xlm-roberta-large-finetuned-conll03-english")


def example_5_1():
    happy_toc = HappyTokenClassification(model_type="BERT", model_name="dslim/bert-base-NER")
    result = happy_toc.classify_token("My name is Geoffrey and I live in Toronto")
    print(type(result))  # <class 'list'>
    print(result[0].word)  # Geoffrey
    print(result[0].entity)  # B-PER
    print(result[0].score)  # 0.9988969564437866
    print(result[0].index)  # 4
    print(result[0].start)  # 11
    print(result[0].end)  # 19
    print(result[1].word)  # Toronto
    print(result[1].entity)  # B-LOC


if __name__ == "__main__":
    # example_5_0()
    example_5_1()