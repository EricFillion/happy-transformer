from happytransformer import HappyWordPrediction


def example_1_0():
    happy_wp_distilbert = HappyWordPrediction()  # default
    happy_wp_albert = HappyWordPrediction("ALBERT", "albert-base-v2")
    happy_wp_bert = HappyWordPrediction("BERT", "bert-base-uncased")
    happy_wp_roberta = HappyWordPrediction("ROBERTA", "roberta-base")


def example_1_1():
    happy_wp = HappyWordPrediction()  # default uses distilbert-base-uncased
    result = happy_wp.predict_mask("I think therefore I [MASK]")
    print(type(result))  # <class 'list'>
    print(result)  # [WordPredictionResult(token_str='am', score=0.10172799974679947)]
    print(type(result[0]))  # <class 'happytransformer.happy_word_prediction.WordPredictionResult'>
    print(result[0])  # [WordPredictionResult(token_str='am', score=0.10172799974679947)]
    print(result[0].token_str)  # am
    print(result[0].score)  # 0.10172799974679947


def example_1_2():
    happy_wp = HappyWordPrediction("ALBERT", "albert-xxlarge-v2")
    result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", top_k=10)
    print(result)  # [WordPredictionResult(token_str='infrastructure', score=0.09270179271697998), WordPredictionResult(token_str='healthcare', score=0.07219093292951584)]
    print(result[1]) # WordPredictionResult(token_str='healthcare', score=0.07219093292951584)
    print(result[1].token_str) # healthcare


def example_1_3():
    happy_wp = HappyWordPrediction("ALBERT", "albert-xxlarge-v2")
    targets = ["technology", "healthcare"]
    result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", targets=targets)
    print(result)  # [WordPredictionResult(token_str='healthcare', score=0.07219093292951584), WordPredictionResult(token_str='technology', score=0.032044216990470886)]
    print(result[1])  # WordPredictionResult(token_str='technology', score=0.032044216990470886)
    print(result[1].token_str)  # technology


def example_1_4():
    happy_wp = HappyWordPrediction("ROBERTA", "roberta-base")
    result = happy_wp.predict_mask("To better the world I would invest in <mask> and education.", top_k=10)
    print(result)   # [WordPredictionResult(token_str='science', score=0.308580607175827), WordPredictionResult(token_str='research', score=0.18018342554569244)]
    print(result[1])    # WordPredictionResult(token_str='research', score=0.18018342554569244)
    print(result[1].token_str)  # research


def main():
    # example_1_1()
    # example_1_1()
    # example_1_2()
    # example_1_3()
    example_1_4()


if __name__ == "__main__":
    main()
