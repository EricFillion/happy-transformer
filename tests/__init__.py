from happytransformer import (
                    HappyGeneration,
                    HappyNextSentence,
                    HappyQuestionAnswering,
                    HappyTextClassification,
                    HappyTokenClassification,
                    HappyTextToText,
                    HappyWordPrediction
                    )


happy_gen =  HappyGeneration("GPT-2", "gpt2")

happy_ns = HappyNextSentence("BERT", "bert-base-uncased")

happy_qa =  HappyQuestionAnswering("DISTILBERT", "distilbert-base-cased-distilled-squad")

happy_tc = HappyTextClassification(model_type="DISTILBERT", model_name="distilbert-base-uncased-finetuned-sst-2-english")

happy_tc_3 = HappyTextClassification(
    model_type="BERT",
    model_name="bert-base-uncased",
    num_labels=3)

happy_toc  = HappyTokenClassification(model_type="BERT", model_name="dslim/bert-base-NER")

happy_tt =  HappyTextToText("T5", "t5-small")

happy_wp =  HappyWordPrediction('BERT', 'bert-base-uncased')


