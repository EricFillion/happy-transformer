"""
Contains the HappyGeneration class
"""

from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.toc.trainer import TOCTrainer
from happytransformer.adaptors import get_adaptor


@dataclass
class GenerationResult:
    text: str


class HappyGeneration(HappyTransformer):
    """
    A user facing class for text generation
    """

    DEFAULT_SETTINGS = {
        "do_sample": False,
        "early_stopping": False,
        "num_beams": 1,
        "temperature": 0.65,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1,
        "length_penalty": 1,
        "no_repeat_ngram_size": 2,
        'bad_words_ids': None,
    }

    def __init__(self, model_type: str = "GPT2", model_name: str = "gpt2"):

        self.adaptor = get_adaptor(model_type)

        model = AutoModelForCausalLM.from_pretrained(model_name)

        super().__init__(model_type, model_name, model)

        self._trainer = TOCTrainer(self.model, model_type, self.tokenizer, self._device, self.logger)

    def __check_gen_text_is_val(self, text):

        if not isinstance(text, str):
            self.logger.error("Please enter a int for the max_length parameter")
            return False
        elif not text:
            self.logger.error("The text input must have at least one character")
            return False
        return True


    def generate_text(self, text, method="greedy", custom_settings=None,
                      min_length=20, max_length=60) -> GenerationResult:
        """
        :param text: starting text that the model uses to generate text with.
        :param method: either one of 1/5 preconfigured methods, or "custom" to indicate custom settings
        :param custom_settings: if method == "custom", then custom settings may be provided in the form of
              a dictionary. Refer to the README to see potential parameters to add.
              Parameters that are not added to the dictionary will keep their default value.
        :return: Text that the model generates.
        """

        is_valid = self.__check_gen_text_is_val(text)

        if is_valid:
            settings = self.get_settings(method, custom_settings)
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            adjusted_min_length = min_length + len(input_ids[0])
            adjusted_max_length = max_length + len(input_ids[0])
            output = self.model.generate(input_ids,
                                         min_length=adjusted_min_length,
                                         max_length=adjusted_max_length,
                                         do_sample=settings['do_sample'],
                                         early_stopping=settings['early_stopping'],
                                         num_beams=settings['num_beams'],
                                         temperature=settings['temperature'],
                                         top_k=settings['top_k'],
                                         top_p=settings['top_p'],
                                         repetition_penalty=settings['repetition_penalty'],
                                         length_penalty=settings['length_penalty'],
                                         no_repeat_ngram_size=settings['no_repeat_ngram_size'],
                                         bad_words_ids=settings['bad_words_ids'],
                                         )
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            final_result = self.__gt_post_processing(result, text)

            return GenerationResult(text=final_result)

        else:
            return GenerationResult(text="")


    def __gt_post_processing(self, result, text):
        """
        A method for processing the output of the model. More features will be added later.
        :param result: result the output of the model after being decoded
        :param text:  the original input to generate_text
        :return: returns to text after going through post-processing. Removes starting text
        """

        return result[len(text):]


    def __get_greedy_settings(self):
        return self.DEFAULT_SETTINGS.copy()

    def __get_beam_settings(self):
        settings = self.DEFAULT_SETTINGS.copy()
        settings["num_beams"] = 5
        settings["early_stopping"] = True
        return settings

    def __get_generic_sampling_settings(self):
        settings = self.DEFAULT_SETTINGS.copy()
        settings["do_sample"] = True
        settings["top_k"] = 0
        settings['temperature'] = 0.7
        return settings

    def __get_top_k_sampling_settings(self):
        settings = self.DEFAULT_SETTINGS.copy()
        settings["do_sample"] = True
        settings['top_k'] = 50
        return settings

    def __get_p_nucleus_sampling_settings(self):
        settings = self.DEFAULT_SETTINGS.copy()
        settings["do_sample"] = True
        settings['top_p'] = 0.92
        settings['top_k'] = 0
        return settings

    def get_settings(self, method, custom_settings):
        settings = {}
        if method == "greedy":
            settings = self.__get_greedy_settings()
        elif method == "beam-search":
            settings = self.__get_beam_settings()
        elif method == "generic-sampling":
            settings = self.__get_generic_sampling_settings()
        elif method == "top-k-sampling":
            settings = self.__get_top_k_sampling_settings()
        elif method == "top-p-nucleus-sampling":
            settings = self.__get_p_nucleus_sampling_settings()
        elif method == "custom":
            settings = self.get_custom_settings(custom_settings)

        return settings

    def get_custom_settings(self, custom_settings):

        possible_keys = list(self.DEFAULT_SETTINGS.keys())
        settings = self.DEFAULT_SETTINGS.copy()

        for key, value in custom_settings.items():

            if key in possible_keys:
                settings[key] = value
            elif key == "min_length":
                self.logger.warning("\"min_length\" is now parameters for the generate_text method")
            elif key == "max_length":
                self.logger.warning("\"max_length\" is now parameters for the generate_text method")
            else:
                self.logger.warning("\"%s\" is not a valid argument", key)
        return settings

    def train(self, input_filepath, args):
        raise NotImplementedError("train() is currently not available")

    def eval(self, input_filepath):
        raise NotImplementedError("eval() is currently not available")

    def test(self, input_filepath):
        raise NotImplementedError("test() is currently not available")
