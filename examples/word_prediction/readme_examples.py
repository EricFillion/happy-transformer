from happytransformer import HappyWordPrediction, WPTrainArgs, WPEvalArgs


def example_4_0():
    happy_wp_distilbert = HappyWordPrediction()  # default
    happy_wp_albert = HappyWordPrediction("ALBERT", "albert-base-v2")
    happy_wp_bert = HappyWordPrediction("BERT", "bert-base-uncased")
    happy_wp_roberta = HappyWordPrediction("ROBERTA", "roberta-base")


def example_4_1():
    happy_wp = HappyWordPrediction()  # default uses distilbert-base-uncased
    result = happy_wp.predict_mask("I think therefore I [MASK]")
    print(type(result))  # <class 'list'>
    print(result)  # [WordPredictionResult(token='am', score=0.10172799974679947)]
    print(type(result[0]))  # <class 'happytransformer.happy_word_prediction.WordPredictionResult'>
    print(result[0])  # [WordPredictionResult(token='am', score=0.10172799974679947)]
    print(result[0].token)  # am
    print(result[0].score)  # 0.10172799974679947


def example_4_2():
    happy_wp = HappyWordPrediction()
    result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", top_k=2)
    print(result)  # [WordPredictionResult(token='health', score=0.1280556619167328), WordPredictionResult(token='science', score=0.07976455241441727)]
    print(result[1])  # WordPredictionResult(token='science', score=0.07976455241441727)
    print(result[1].token)  # science

def example_4_3():
    happy_wp = HappyWordPrediction()
    targets = ["technology", "healthcare"]
    result = happy_wp.predict_mask("To better the world I would invest in [MASK] and education.", targets=targets, top_k=2)
    print(result)  # [WordPredictionResult(token='healthcare', score=0.07380751520395279), WordPredictionResult(token='technology', score=0.009395276196300983)]
    print(result[1])  # WordPredictionResult(token='technology', score=0.009395276196300983)
    print(result[1].token)  # technology

def example_4_4():
    happy_wp = HappyWordPrediction()
    args = WPTrainArgs(num_train_epochs=1)
    happy_wp.train("../../data/wp/train-eval.txt", args=args)

def example_4_5():
    happy_wp = HappyWordPrediction()
    args = WPEvalArgs(preprocessing_processes=2)
    result = happy_wp.eval("../../data/wp/train-eval.txt", args=args)
    print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
    print(result)  # EvalResult(eval_loss=0.459536075592041)
    print(result.loss)  # 0.459536075592041


def main():
    # example_4_0()
    # example_4_1()
    #example_4_2()
    # example_4_3()
    # example_4_4()
    example_4_5()




if __name__ == "__main__":
    main()
