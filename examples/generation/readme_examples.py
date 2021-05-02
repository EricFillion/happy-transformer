from happytransformer import HappyGeneration


def example_1_0():
    happy_gen = HappyGeneration("GPT2", "gpt2")  # default
    # happy_gen = HappyGeneration("GPT2", "gpt2-large")  # Good for Google Colab
    # happy_gen = HappyGeneration("GPT2", "gpt2-xl")  # Best performance

def example_1_1():
    happy_wp = HappyGeneration()  # default uses distilbert-base-uncased
    result = happy_wp.generate_text("artificial intelligence is ", max_length=15)
    print(result)  # GenerationResult(text='\xa0a new field of research that has been gaining momentum in recent years.')
    print(result.text)  #  a new field of research that has been gaining momentum in recent years.

if __name__ == "__main__":
    # example_1_0()
    example_1_1()

