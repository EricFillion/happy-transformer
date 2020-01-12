"""
An example of how to use a HappyTRANSFORMER to create a
that gives instant feedback on written sentences
"""

import random
from happytransformer import HappyROBERTA

class TeachingTransformer:
    """
    A class that organizes and runs a program that makes a game out of learning English
    """

    no_space_characters = ['.', ',', '?', '!'] # characters that do not require a space before being used

    def __init__(self, transformer):
        """
        :param transformer: A HappyTransformer object
        """
        self.transformer = transformer

    def __predict_last_word(self, text: str):
        """
        :param text: An incomplete sentence
        :return: 1. A list with str word predictions
                 2. A list with the corresponding softmax for each prediction.
                 Both lists share the same indexes for each prediction.
        """
        text = text.strip()
        text = text + " [MASK]"
        predictions = self.transformer.predict_mask(text, num_results=400)

        word_predictions = list()
        score_predictions = list()
        special_characters = [".", ",", "?", "!", "$", "*"]  # TODO add more special characters

        count = 0
        for prediction in predictions:

            word = prediction["word"]
            word = word.strip()

            if word.isalpha() == True or word in special_characters:
                word_predictions.append(word)
                score_predictions.append(prediction["softmax"])
                count += 1
                if count == 200:
                    break

        return word_predictions, score_predictions

    def teach(self):
        """
        Runs a terminal program for Teaching Transformer.
        The user must enter one word at a time.
        If the user enters the word "$clear," then the program
        generates a new starting phrase.
        If the user enters the word "$exit,"  then the terminal program
        exits.
        """
        sample_starts = ["To solve world hunger we must", "Humans are for", "I think therefore I"]
        text = random.choice(sample_starts)

        while True:
            word_predictions, score_predictions = self.__predict_last_word(text)
            new_word = input(text + " ")
            new_word = new_word.strip()

            # TODO check for valid input
            if new_word == "$clear":
                text = random.choice(sample_starts)
            elif new_word == "$exit":
                print("Exiting Teaching Transformer")
                break

            else:
                if new_word in self.no_space_characters:
                    text = text + new_word

                else:
                    text = text + " " + new_word

                top_word_predictions = word_predictions[:5]
                prediction_index = 201  # default prediction index

                if new_word in word_predictions:
                    prediction_index = word_predictions.index(new_word)
                    prediction_score = score_predictions[prediction_index]
                else:
                    prediction_score = 0  # new_word is not in the top 200 predictions

                if prediction_index == 0:
                    feedback = "PERFECT"
                elif prediction_index < 5:
                    feedback = "amazing"
                elif prediction_index < 8:
                    feedback = "great"
                elif prediction_score > 0.05:
                    feedback = "good"
                elif prediction_score > 0.03:
                    feedback = "okay"
                elif prediction_score > 0.02:
                    feedback = 'Okay, but could be better'
                elif prediction_score > 0.01:
                    feedback = "So so, there is room for improvement"
                else:
                    feedback = "Nice try, look at the examples for better options"

                print(feedback)
                print("Here are some suggestions: ", end="")
                print(top_word_predictions)
                print("\n\n")


def main():
    happy_roberta = HappyROBERTA('roberta-large')

    teching_transformer = TeachingTransformer(happy_roberta)

    teching_transformer.teach()


if __name__ == "__main__":
     main()

