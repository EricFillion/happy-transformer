from happytransformer import HappyTextToText, TTSettings, TTTrainArgs, TTEvalArgs

def example_7_0():
    # --------------------------------------#
    happy_tt = HappyTextToText("T5", "t5-small")  # default

def example_7_1():
    # --------------------------------------#
    happy_tt = HappyTextToText()  # default uses t5-small
    top_p_sampling_settings = TTSettings(do_sample=True, top_k=0, top_p=0.8, temperature=0.7,  min_length=20, max_length=20, early_stopping=True)
    result = happy_tt.generate_text("translate English to French: nlp is a field of artificial intelligence", args=top_p_sampling_settings)
    print(result)  # nlp est un domaine de l’intelligence artificielle. n
    print(result.text)  # nlp est un domaine de l’intelligence artificielle. n


def example_7_2():
    happy_tt = HappyTextToText("T5", "t5-small")

    greedy_settings = TTSettings(no_repeat_ngram_size=2, max_length=20)
    output_greedy = happy_tt.generate_text(
        "translate English to French: nlp is a field of artificial intelligence ",
        args=greedy_settings)

    beam_settings = TTSettings(num_beams=5, max_length=20)
    output_beam_search = happy_tt.generate_text(
        "translate English to French: nlp is a field of artificial intelligence ",
        args=beam_settings)

    generic_sampling_settings = TTSettings(do_sample=True, top_k=0, temperature=0.7, max_length=20)
    output_generic_sampling = happy_tt.generate_text(
        "translate English to French: nlp is a field of artificial intelligence ",
        args=generic_sampling_settings)

    top_k_sampling_settings = TTSettings(do_sample=True, top_k=50, temperature=0.7, max_length=20)
    output_top_k_sampling = happy_tt.generate_text(
        "translate English to French: nlp is a field of artificial intelligence ",
        args=top_k_sampling_settings)

    top_p_sampling_settings = TTSettings(do_sample=True, top_k=0, top_p=0.8, temperature=0.7, max_length=20)
    output_top_p_sampling = happy_tt.generate_text(
        "translate English to French: nlp is a field of artificial intelligence ",
        args=top_p_sampling_settings)

    print("Greedy:", output_greedy.text)  # Greedy: nlp est un domaine de l'intelligence artificielle
    print("Beam:", output_beam_search.text)  # Beam: nlp est un domaine de l'intelligence artificielle
    print("Generic Sampling:", output_generic_sampling.text)  # Generic Sampling: nlp est un champ d'intelligence artificielle
    print("Top-k Sampling:", output_top_k_sampling.text)  # Top-k Sampling: nlp est un domaine de l’intelligence artificielle
    print("Top-p Sampling:", output_top_p_sampling.text)  # Top-p Sampling: nlp est un domaine de l'intelligence artificielle

def example_7_3():
    happy_tt = HappyTextToText()
    args = TTTrainArgs(num_train_epochs=1)
    happy_tt.train("../../data/tt/train-eval-grammar.csv", args=args)

def example_7_4():
    happy_tt = HappyTextToText()
    args = TTEvalArgs(preprocessing_processes=1)
    result = happy_tt.eval("../../data/tt/train-eval-grammar.csv", args=args)
    print(type(result))  # <class 'happytransformer.happy_trainer.EvalResult'>
    print(result)  # EvalResult(loss=3.2277376651763916)
    print(result.loss)  # 3.2277376651763916


def main():
    # example_7_0()
    #example_7_1()
    # example_7_2()
    # example_7_3()
    example_7_4()

if __name__ == "__main__":
    main()
