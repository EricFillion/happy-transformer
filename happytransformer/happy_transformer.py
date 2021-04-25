"""
Contains the parent class to HappyTextClassification, HappyWordPrediction, HappyQuestionAnswering
and HappyNextSentencePrediction called HappyTransformer

Contains shared variables and methods for these classes.
"""
import logging
import torch
from transformers import  AutoTokenizer

class HappyTransformer():
    """
    Parent class to HappyTextClassification, HappyWordPrediction, HappyQuestionAnswering
    and HappyNextSentencePrediction.

    """

    def __init__(self, model_type, model_name, model):
        self.model_type = model_type  # BERT, #DISTILBERT, ROBERTA, ALBERT etc
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.model.eval()
        self._trainer = None  # initialized in child class

        # todo  change logging system
        self.logger = logging.getLogger(__name__)

        handler = logging.StreamHandler()
        handler.addFilter(logging.Filter('happytransformer'))
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[handler]
        )

        self._device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        if self._device == 'cuda':
            self.model.to(self._device)
        self.logger.info("Using model: %s", self._device)


    def train(self, input_filepath, args):
        """
        Trains a model
        :param input_filepath: a string that contains a path to a csv file
         that contains testing data
        :param args: settings in the form of a dictionary
        :return: None
        """
        raise NotImplementedError()

    def eval(self, input_filepath, args):
        """
        Evaluates the model. Determines how well the model performs on a given dataset
        :param input_filepath: a string that contains a path to a
         csv file that contains evaluating data
        :param args: settings in the form of a dictionary
        :return: correct percentage
        """
        raise NotImplementedError()

    def test(self, input_filepath, args):
        """
        Used to generate predictions for a given dataset.
        The dataset may not be labelled.
        :param args: settings in the form of a dictionary

        :param input_filepath: a string that contains a path to
        a csv file that contains testing data

        """
        raise NotImplementedError()

    def _create_args_dataclass(self, default_dic_args, input_dic_args, method_dataclass_args):
        """
        Combines default_dic_args and input_dic_args and then outputs a dataclass.

        The values of input_dic_args overwrite the values of default_dic_args.

        default_dic_args and dataclass_args must have the same keys.

        :param default_dic_args: A dictionary that contains default settings. Example: ARGS_WP_EVAl
        :param input_dic_args: A dictionary a user inputs for **kwargs when using .train(), .eval() or .test()
        :param method_dataclass_args: A dataclass object for target functionality. Example: WPEvalArgs
        :return: A dataclass object that will then be passed to HappyTrainer.train()/eval/test
        """
        settings_dic = {**default_dic_args, **input_dic_args}
        return method_dataclass_args(**settings_dic)
