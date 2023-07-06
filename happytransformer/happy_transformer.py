"""
Contains the parent class to HappyTextClassification, HappyWordPrediction, HappyQuestionAnswering
and HappyNextSentencePrediction called HappyTransformer

Contains shared variables and methods for these classes.
"""
import logging
import torch
from transformers import  AutoTokenizer, AutoConfig
from happytransformer.happy_trainer import  TrainArgs
from happytransformer.fine_tuning_util import create_args_dataclass

class HappyTransformer():
    """
    Parent class to HappyTextClassification, HappyWordPrediction, HappyQuestionAnswering
    and HappyNextSentencePrediction.

    """

    def __init__(self, model_type, model_name, model, load_path="", use_auth_token: str = None):
        self.model_type = model_type  # BERT, #DISTILBERT, ROBERTA, ALBERT etc
        self.model_name = model_name

        if load_path != "":
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
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

        self.device = None

        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                self.device = torch.device("mps")

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        if not self.device:
            self.device = torch.device("cpu")

        if self.device.type != 'cpu':
            self.model.to(self.device)

        self.logger.info("Using model: %s", self.device)


    def train(self, input_filepath: str ,  args: TrainArgs, eval_filepath: str = "", ):
        """
        Trains a model
        :param input_filepath: A string that contains a path to a file that contains training data.
        :param input_filepath: A  string that contains a path to a file that contains eval data.
        :param args: A TrainArgs() child class such as GENTrainArgs()
        :return: None
        """
        if type(args) == dict:
            raise ValueError("Dictionary training arguments are no longer supported as of Happy Transformer version 2.5.0.")


        self._trainer.train(input_filepath=input_filepath, eval_filepath=eval_filepath,
                            dataclass_args=args)



    def eval(self, input_filepath, args):
        """
        Evaluates the model. Determines how well the model performs on a given dataset
        :param input_filepath: a string that contains a path to a
         csv file that contains evaluating data
        :param args: settings in the form of a dictionary
        :return: correct percentage
        """
        if type(args) == dict:
            raise ValueError("Dictionary evaluating arguments are no longer supported as of Happy Transformer version 2.5.0.")

        return self._trainer.eval(input_filepath=input_filepath, dataclass_args=args)


    def test(self, input_filepath, args):
        """
        Used to generate predictions for a given dataset.
        The dataset may not be labelled.
        :param args: settings in the form of a dictionary

        :param input_filepath: a string that contains a path to
        a csv file that contains testing data

        """
        raise NotImplementedError()

    def save(self, path):
        """
        Saves both the model, tokenizer and various configuration/settings files
        to a given path

        :param path: string:  a path to a directory
        :return:
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

