from happytransformer import HappyGeneration, GENSettings, GENTrainArgs, GENEvalArgs

def example_1_0():
    happy_gen = HappyGeneration("GPT2", "gpt2")  # default
    # happy_gen = HappyGeneration("GPT2", "gpt2-large")  # Good for Google Colab
    # happy_gen = HappyGeneration("GPT2", "gpt2-xl")  # Best performance

def example_1_1():
    happy_wp = HappyGeneration()  # default uses distilbert-base-uncased
    result = happy_wp.generate_text("artificial intelligence is ", max_length=15)
    print(result)  # GenerationResult(text='\xa0a new field of research that has been gaining momentum in recent years.')
    print(result.text)  #  a new field of research that has been gaining momentum in recent years.

def example_1_2():
    happy_gen = HappyGeneration()

    greedy_settings = GENSettings(no_repeat_ngram_size=2)
    output_greedy = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=greedy_settings, min_length=5, max_length=5)

    beam_settings = GENSettings(early_stopping=True, num_beams=5)
    output_beam_search = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=beam_settings, min_length=5, max_length=5)

    generic_sampling_settings = GENSettings(do_sample=True, early_stopping=False, top_k=0, temperature=0.7)
    output_generic_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=generic_sampling_settings, min_length=5, max_length=5)

    top_k_sampling_settings = GENSettings(do_sample=True, early_stopping=False, top_k=50, temperature=0.7)
    output_top_k_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=top_k_sampling_settings, min_length=5, max_length=5)


def example_1_3():
    happy_gen = HappyGeneration()
    args = GENTrainArgs(num_train_epochs=1)
    happy_gen.train("../../data/gen/train-eval.txt", args=args)

def example_1_4():
    happy_gen = HappyGeneration()
    args = GENEvalArgs(preprocessing_processes=2)
    result = happy_gen.eval("../../data/gen/train-eval.txt", args=args)
    print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
    print(result)  # EvalResult(loss=3.3437771797180176)
    print(result.loss)  # 3.3437771797180176


if __name__ == "__main__":
    example_1_0()
    # example_1_1()
    # example_1_2()
    # example_1_3()
    # example_1_4()


