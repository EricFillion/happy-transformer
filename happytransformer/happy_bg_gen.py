"""
contains HappyBackgroundGeneration class
"""

from dataclasses import dataclass
from happytransformer.happy_generation import gen_greedy_settings

@dataclass
class BackgroundGenerationResult:
    text: str


class HappyBackgroundGeneration:

    def __init__(self, happy_gen, nlp=None):
        """
        :param happy_gen: a HappyTextGeneration object
        :param nlp: an optional spacy nlp object to improve performance
        """
        self.happy_gen = happy_gen
        self.nlp = nlp

    def generate_background_info(self, target, context="", settings = gen_greedy_settings,
                                 min_length=10, max_length=30) -> BackgroundGenerationResult:
        """
        Generates background information for a given noun
        :param target: A single noun to generate background information for
        :param context: A string that provides context to help the model generate information
        :param method: either one of 1/5 preconfigured methods, or "custom" to indicate custom settings
        :param custom_settings: if method == "custom", then custom settings may be provided in the form of
              a dictionary. Refer to the README to see potential parameters to add.
              Parameters that are not added to the dictionary will keep their default value.
        :param min_length: minimal number of returned tokens (as a combined string)
        :param max_length: maximum number of returned tokens (as a combined string)
        :return: a BackgroundGenerationResult object. Contains a single parameter: text
        """
        question = self._get_question(target)
        prompt = self._get_prompt(context, question)

        output = self.happy_gen.generate_text(prompt, settings=settings,
                                              min_length=min_length, max_length=max_length)
        return BackgroundGenerationResult(
            text=output.text,
        )

    def _get_prompt(self, context, question):
        """

        :param context: string: information to help the model understand the topic
        :param question: string: the question that the model will attempt answer
        :return: string: concatenates context and question while dealing with special cases
        """
        if context[-1:] == ".":
            return context + " " + question

        else:
            return context + ". " + question

    def _get_question(self, target) -> str:
        """
        :param: string: target:
        :return: string: generates a question for the model to answer
        """
        if self.nlp is not None:
            doc = self.nlp(target)
            if doc[0].tag_ == 'NNS':
                return f"What are {target}?"
            if doc.ents:
                if doc.ents[0].label_ == "PERSON":
                    return f"Who is {target}?"

        return f"What is a {target}?"
