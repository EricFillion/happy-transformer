"""
Contains the HappyGeneration class
"""
from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.adaptors import get_adaptor

"""
The main settings that users will adjust when performing experiments

They may still modify all of the settings found here:
    https://huggingface.co/transformers/main_classes/model.html#generation.
    
The values for full_settings are the same as the default values above except for min and max length. 
"""
full_settings = {
    "do_sample": False,
    "early_stopping": False,
    "num_beams": 1,
    "temperature": 1,
    "top_k": 50,
    "no_repeat_ngram_size": 0,
}

# greedy is prone to repetition loops. So, we'll set no_repeat_ngram_size to 2.
default_greedy_settings = {
    "do_sample": False,
    "early_stopping": False,
    "no_repeat_ngram_size": 2,
}

default_beam_settings = {
    "do_sample": False,
    "early_stopping": True,
    "num_beams": 5,
}

default_generic_sampling_settings = {
    "do_sample": True,
    "early_stopping": False,
    "top_k": 0,
    "temperature": 0.7,
}


default_top_k_sampling_settings = {
    "do_sample": True,
    "early_stopping": False,
    "top_k": 50,
    "temperature": 0.7,
}


@dataclass
class GenerationResult:
    text: str


class HappyGeneration(HappyTransformer):
    """
    A user facing class for text generation
    """
    def __init__(self, model_type: str = "GPT2", model_name: str = "gpt2"):

        self.adaptor = get_adaptor(model_type)

        model = AutoModelForCausalLM.from_pretrained(model_name)

        super().__init__(model_type, model_name, model)

        self._trainer = None

    def __assert_default_text_is_val(self, text):

        if not isinstance(text, str):
            raise ValueError("The text input must be a string")
        if not text:
            raise ValueError("The text input must have at least one character")


    def generate_text(self, text, settings=default_beam_settings,
                      min_length=20, max_length=60) -> GenerationResult:
        """
        :param text: starting text that the model uses to generate text with.
        :param settings: A dictionary that contains settings that determine what algorithm is used to generate text
        :param min_length: The minimum number of tokens for the output
        :param max_length: The maximum number of tokens for the output
        :return: Text that the model generates.
        """

        self.__assert_default_text_is_val(text)
        self.__validate_settings(settings, min_length, max_length)

        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        adjusted_min_length = min_length + len(input_ids[0])
        adjusted_max_length = max_length + len(input_ids[0])

        output = self.model.generate(input_ids,
                                     **settings,
                                     min_length=adjusted_min_length,
                                     max_length=adjusted_max_length,
                                     )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        final_result = self.__gt_post_processing(result, text)

        return GenerationResult(text=final_result)



    def __gt_post_processing(self, result, text):
        """
        A method for processing the output of the model. More features will be added later.
        :param result: result the output of the model after being decoded
        :param text:  the original input to generate_text
        :return: returns to text after going through post-processing. Removes starting text
        """

        return result[len(text):]

    def __validate_settings(self, settings, min_length, max_length):
        """
        The default function for min_length and max_length within Hugging Face is for the
        min/max  number of tokens of the combined input/output .

        We believe it makes more sense for them to be just for the output.

        So, we added custom min_length and max_length to the generate function.

        We must ensure that the user does not include min_length/max_length within the input dictionary
        """
        if "min_length" in settings:
            self.logger.warning("\"min_length\" is a parameter for the method \"generate_text\"."
                                " Please use this parameter instead of adding it to the settings dictionary. "
                                "%s, is being used. It represents the minimum number of tokens for the output",
                                str(min_length))
            del settings["min_length"]

        if "max_length" in settings:
            self.logger.warning("\"max_length\" is a parameter for the method \"generate_text\"."
                                " Please use this parameter instead of adding it to the settings dictionary. "
                                "%s is being used. It represents the  maximum number of tokens for the output",
                                str(max_length))
            del settings["max_length"]


    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
