from happytransformer import HappyGeneration, GENSettings, GENTrainArgs, GENEvalArgs

def example_1_0():
    happy_gen = HappyGeneration("GPT2", "gpt2")  # default
    # happy_gen = HappyGeneration("GPT2", "gpt2-large")  # Good for Google Colab
    # happy_gen = HappyGeneration("GPT2", "gpt2-xl")  # Best performance

def example_1_1():
    happy_gen = HappyGeneration()  # default uses distilbert-base-uncased
    args = GENSettings(max_length=15)
    result = happy_gen.generate_text("artificial intelligence is ", args=args)
    print(result)  # GenerationResult(text='\xa0a new field of research that has been gaining momentum in recent years.')
    print(result.text)  #  a new field of research that has been gaining momentum in recent years.

def example_1_2():
    happy_gen = HappyGeneration()

    greedy_settings = GENSettings(no_repeat_ngram_size=2,  max_length=10)
    output_greedy = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=greedy_settings)

    beam_settings = GENSettings(early_stopping=True, num_beams=5,  max_length=10)
    output_beam_search = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=beam_settings)

    generic_sampling_settings = GENSettings(do_sample=True, early_stopping=False, top_k=0, temperature=0.7,  max_length=10)
    output_generic_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=generic_sampling_settings)

    top_k_sampling_settings = GENSettings(do_sample=True, early_stopping=False, top_k=50, temperature=0.7,  max_length=10)
    output_top_k_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=top_k_sampling_settings)

    top_p_sampling_settings = GENSettings(do_sample=True, top_k=0, top_p=0.8, temperature=0.7, max_length=10)
    output_top_p_sampling = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=top_p_sampling_settings)

    print("Greedy:", output_greedy.text)  # a new field of research that has been gaining
    print("Beam:", output_beam_search.text)  # one of the most promising areas of research in
    print("Generic Sampling:", output_generic_sampling.text)  #  an area of highly promising research, and a
    print("Top-k Sampling:", output_top_k_sampling.text)  # a new form of social engineering. In this
    print("Top-p Sampling:", output_top_p_sampling.text)  # a new form of social engineering. In this


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

def example_1_5():
    happy_gen = HappyGeneration()
    args = GENSettings(bad_words = ["new form", "social"]) # Provide a list of bad words/phrases
    result = happy_gen.generate_text(
        "Artificial intelligence is ",
        args=args)
    print(result.text)
    




if __name__ == "__main__":
    # example_1_0()
    # example_1_1()
    example_1_2()
    # example_1_3()
    # example_1_4()
    # example_1_5()


