# disable pylint TODO warning
# pylint: disable=W0511
# pylint: disable=C0301

"""
HappyTransformer is a wrapper over pytorch_transformers to make it
easier to use.
"""

from collections import namedtuple
import string
import re
import os
import sys
import csv
import logging
import logging.config
import numpy as np
import torch
import pandas as pd

from happytransformer.classifier_args import classifier_args
from happytransformer.sequence_classifier import SequenceClassifier
from happytransformer.mlm_utils import FinetuneMlm, word_prediction_args
from happytransformer.sentence import is_one_sentence

_POSSIBLE_MASK_TOKENS = ['<mask>', '<MASK>', '[MASK]']

class HappyTransformer:
    """
    Initializes pytroch's transformer models and provided methods for
    their basic functionality.
    Philosophy: Automatically make decisions for the user so that they don't
                have to have any understanding of PyTorch or transformer
                models to be able to utilize their capabilities.
    """

    def __init__(self, model, model_name):
        # Transformer and tokenizer set in child class
        self.model = model
        self.model_name = model_name
        self.mlm = None  # Masked Language Model
        self.seq = None  # Sequence Classification
        self.qa = None  # Question Answering
        self.mlm_args = None  # Mask Language Model Finetuning
        self.tokenizer = None

        # GPU support
        self.gpu_support = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # show only happytransformer logs
        handler = logging.StreamHandler()
        handler.addFilter(logging.Filter('happytransformer'))
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[handler]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info("Using model: %s", self.gpu_support)
        self.seq_trained = False
        self.mwp_trainer = None
        self.mwp_trained = False

    def _get_masked_language_model(self):
        raise NotImplementedError()

    def _get_next_sentence_prediction(self):
        raise NotImplementedError()

    def _get_question_answering(self):
        raise NotImplementedError()

    def _standardize_mask_tokens(self, text):
        '''
        convert mask tokens to mask token preferred by tokenizer
        '''
        for possible_mask_token in _POSSIBLE_MASK_TOKENS:
            text = text.replace(possible_mask_token, self.tokenizer.mask_token)
        return text

    def _prepare_mlm(self):
        if self.mlm is None:
            self._get_masked_language_model()
        if self.gpu_support=='cuda':
            self.mlm.to('cuda')

    def _masked_predictions_at_index_any(self, softmax, index, k):
        '''
        return top predictions for a mask token from all embeddings
        '''
        scores_tensor, token_ids_tensor = torch.topk(softmax[index], k)
        scores = scores_tensor.tolist()
        token_ids = token_ids_tensor.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        options = [
            self._postprocess_option(token)
            for token in tokens
        ]
        return [
            {"word": option, "softmax": score}
            for option, score in zip(options, scores)
        ]

    def _masked_predictions_at_index_options(self, softmax, index, options):
        '''
        return top predictions for a mask token from a list of options
        '''
        option_ids = [
            self.tokenizer.encode(option) 
            for option in options
        ]
        scores = [
            self.soft_sum(option_id, softmax, index)
            for option_id in option_ids
        ]
        return [
            {"word": option, "softmax": score}
            for option,score in zip(options,scores)
        ]

    def _postprocess_option(self, text: str):
        '''
        modifies option text as seen by predict_masks() output.
        override in subclass to filter out weird characters.
        :param text: original text of prediction option
        :returns text: processed text of prediction option
        '''
        return text

    def predict_masks(self, text: str, options=None, num_results=1):
        '''
        Predict multiple [MASK] tokens in some text.
        :param text: text containing the mask tokens
        :param masks_options: list of lists of options as strings
        :param num_results: number of results to return per mask token
        num_results is ignored if options are supplied.
        :returns: A list of list of namedtuples of the form (text,probability),
        where predictions are ordered descendingly by likelihood
        '''
        self._prepare_mlm()
        self._verify_mask_text(text)
        text = self._standardize_mask_tokens(text)

        token_ids = self.tokenizer.encode(text, return_tensors='pt')
        softmax = self._get_prediction_softmax(token_ids)

        masked_indices = [
            idx
            for idx, token_id in enumerate(token_ids[0].tolist())
            if token_id == self.tokenizer.mask_token_id
        ]
        
        if options is None:
            return [
                self._masked_predictions_at_index_any(
                    softmax, masked_index, num_results
                )
                for masked_index in masked_indices
            ]
        else:
            return [
                self._masked_predictions_at_index_options(
                    softmax, masked_index, mask_options
                )
                for masked_index, mask_options in zip(masked_indices, options)
            ]

    def predict_mask(self, text: str, options=None, num_results=1):
        '''
        Predict a single [MASK] token in some text.
        :param text: text containing the mask token
        :param options: list of options as strings
        :param num_results: number of predictions to return if no options supplied
        :returns: list of dictionaries with keys 'word' and 'softmax'
        '''
        masks_options = None if options is None else [options]
        predictions = self.predict_masks(text, masks_options, num_results)
        return self.__format_option_scores(predictions[0])

    def _get_prediction_softmax(self, token_ids):
        """
        Gets the softmaxes of the predictions for each index in the the given
        input string.
        Returned tensor will be in shape:
            [1, <tokens in string>, <possible options for token>]
        :param text: a tokenized string to be used by the transformer.
        :return: a tensor of the softmaxes of the predictions of the
                 transformer

        """

        if self.gpu_support == "cuda":
            token_ids = token_ids.to('cuda')

        with torch.no_grad():
            outputs = self.mlm(token_ids)
            return torch.softmax(outputs.logits[0], dim=-1)

    def __format_option_scores(self, tupled_predicitons: list):
        """
        Formats the given list of tuples containing the option and its
        corresponding softtmax into a user friendly list of dictionaries where
        the first element in the list is the option with the highest softmax.
        Dictionary will be in the form:
             {'word': <the option>, 'softmax': <sofmax for the option>}
        :param: ranked_scores: list of tuples to be converted into user
                friendly dicitonary
        :return: formatted_ranked_scores: list of dictionaries of the ranked
                 scores
        """
        ranked_scores = sorted(tupled_predicitons, key=lambda x: x["softmax"],
                               reverse=True)
        formatted_ranked_scores = list()
        for dic in ranked_scores:

            formatted_ranked_scores.append({'word': dic["word"], 'softmax': dic["softmax"]})
        return formatted_ranked_scores

    def predict_next_sentence(self, sentence_a, sentence_b, use_probability=False):
        """
        Determines if sentence B is likely to be a continuation after sentence
        A.
        :param sentence_a: First sentence
        :param sentence_b: Second sentence to test if it comes after the first
        :param use_probability: Toggle outputting probability instead of boolean
        :return Result of whether sentence B follows sentence A,
                as either a probability or a boolean
        """

        if not is_one_sentence(sentence_a) or not is_one_sentence(sentence_b):
            raise ValueError('Each inputted text variable for the "predict_next_sentence" method must contain a single sentence')

        if self.nsp is None:
            self._get_next_sentence_prediction()

        if self.gpu_support == 'cuda':
            self.nsp.to('cuda')

        encoded = self.tokenizer(sentence_a, sentence_b, return_tensors='pt')
        with torch.no_grad():
            scores = self.nsp(encoded['input_ids'], token_type_ids=encoded['token_type_ids']).logits[0]

        probabilities = torch.softmax(scores, dim=0)
        # probability that sentence B follows sentence A
        correct_probability = probabilities[0].item()

        if self.gpu_support == 'cuda':
            torch.cuda.empty_cache()

        return (
            correct_probability if use_probability else 
            correct_probability >= 0.5
        )

    def _get_segment_ids(self, tokenized_text: list):
        """
        Converts a list of tokens into segment_ids. The segment id is a array
        representation of the location for each character in the
        first and second sentence. This method only words with 1-2 sentences.
        Example:
        tokenized_text = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]',
                          'jim', '[MASK]', 'was', 'a', 'puppet', '##eer',
                          '[SEP]']
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        returns segments_ids
        """
        split_location = tokenized_text.index(self.tokenizer.sep_token)

        segment_ids = [
            0 if idx <= split_location else 1
            for idx in range(len(tokenized_text))
        ]
        # add exception case for XLNet

        return segment_ids

    def _verify_mask_text(self, text: str):

        if all(
            mask_token not in text
            for mask_token in _POSSIBLE_MASK_TOKENS
        ):
            raise ValueError('No mask token found')
        if '[MASK]' not in text:
            self.logger.warn("[MASK] was not found in your string. Change the word you want to predict to [MASK]")
        if '[CLS]' in text:
            raise ValueError("[CLS] was found in your string.  Remove it as it will be automatically added later")
        if '[SEP]' in text:
            raise ValueError("[SEP] was found in your string.  Remove it as it will be automatically added later")

    @staticmethod
    def soft_sum(option: list, softed, mask_id: int):
        # TODO: Better logic.
        """
        Adds the softmax of a single option
        XLNET tokenizer sometimes splits words in to pieces.
        Ex: The councilmen -> ['the', 'council', 'men']
        Pretty sure that this is mathematically wrong
        :param option: Id of tokens in one option
        :param softed: softmax of the output
        :param mask: Index of masked word
        :return: float Tensor
        """
        # Collects the softmax of all tokens in list
        return np.sum([softed[mask_id][op] for op in option])

    def init_sequence_classifier(self):
        """
        Initializes a binary sequence classifier model with default settings
        """

        # TODO Test the sequence classifier with other models
        args = classifier_args.copy()
        self.seq = SequenceClassifier(args, self.tokenizer, self.logger, self.gpu_support, self.model, self.model_name)

        self.logger.info("A binary sequence classifier for %s has been initialized", self.model_name)

    def custom_init_sequence_classifier(self, args):
        """
        Initializes a binary sequence classifier model with custom settings.
        The default settings args dictionary can be found  happy_transformer/sequence_classification/classifier_args.
        This dictionary can then be modified and then used as the only input for this method.

        """
        self.seq = SequenceClassifier(args, self.tokenizer, self.logger, self.gpu_support, self.model, self.model_name)
        self.logger.info("A binary sequence classifier for %s has been initialized", self.model_name)

    def train_sequence_classifier(self, train_csv_path):
        """
        Trains the HappyTransformer's sequence classifier

        :param train_csv_path: A path to the csv evaluation file.
            Each test is contained within a row.
            The first column is for the the correct answers, either 0 or 1 as an int or a string .
            The second column is for the text.
        """
        self.logger.info("***** Running Training *****")

        train_df = self.__process_classifier_data(train_csv_path)

        if self.seq is None:
            raise ValueError("Initialize the sequence classifier before training")

        sys.stdout = open(os.devnull,
                          'w')  # Disable printing to stop external libraries from printing
        train_df = train_df.astype("str")
        self.seq.train_list_data = train_df.values.tolist()
        del train_df  # done with train_df
        self.seq.train_model()
        self.seq_trained = True
        sys.stdout = sys.__stdout__  # Enable printing

    def eval_sequence_classifier(self, eval_csv_path):
        """
        Evaluates the trained sequence classifier against a testing set.

        :param csv_path: A path to the csv evaluation file.
            Each test is contained within a row.
            The first column is for the the correct answers, either 0 or 1 as an int or a string .
            The second column is for the text.

        :return: A dictionary evaluation matrix
        """

        self.logger.info("***** Running evaluation *****")

        sys.stdout = open(os.devnull, 'w')  # Disable printing

        eval_df = self.__process_classifier_data(eval_csv_path)

        if not self.seq_trained:
            raise ValueError("Train the sequence classifier before evaluation")

        eval_df = eval_df.astype("str")
        self.seq.eval_list_data = eval_df.values.tolist()

        results = self.seq.evaluate()
        sys.stdout = sys.__stdout__  # Enable printing

        return results

    def test_sequence_classifier(self, test_csv_path):
        """

        :param test_csv_path: a path to the csv evaluation file.
            Each test is contained within a row.
            The first column is for the the correct answers, either 0 or 1 as an int or a string .
            The second column is for the text.
        :return: A list of predictions where each prediction index is the same as the corresponding test's index
        """
        self.logger.info("***** Running Testing *****")
        sys.stdout = open(os.devnull, 'w')  # Disable printing

        test_df = self.__process_classifier_data(test_csv_path, for_test_data=True)

        if not self.seq_trained:
            raise ValueError("Train the sequence classifier before testing")

        test_df = test_df.astype("str")
        self.seq.test_list_data = test_df.values.tolist()
        del test_df  # done with test_df

        results = self.seq.test()

        sys.stdout = sys.__stdout__  # Enable printing

        return results

    def __process_classifier_data(self, csv_path, for_test_data=False):
        """
         Credit: This code was modified from this repository
         https://github.com/ThilinaRajapakse/pytorch-transformers-classification
        :param csv_path: Path to csv file that must be processed
        :return: A Panda dataframe with the proper information for classification tasks
        """

        if for_test_data:
            with open(csv_path, 'r') as test_file:
                reader = csv.reader(test_file)
                text_list = list(reader)
            # Blank values are required for the first column value the testing data to increase
            # reusability of preprocessing methods between the tasks
            blank_values = ["0"] * len(text_list)
            data_frame = pd.DataFrame([*zip(blank_values, text_list)])
            del blank_values  # done with blank_values

        else:
            data_frame = pd.read_csv(csv_path, header=None)

        data_frame[0] = data_frame[0].astype("int")
        data_frame = pd.DataFrame({
            'id': range(len(data_frame)),
            'label': data_frame[0],
            'alpha': ['a'] * data_frame.shape[0],
            'text': data_frame[1].replace(r'\n', ' ', regex=True)
        })

        return data_frame

    def init_train_mwp(self, args=None):
        """
        Initializes the MLM for fine-tuning on masked word prediction.
        If args are not supplied the following hyperparameters are used:
            batch size = 1
            Number of epochs  = 1
            Learning rate = 5e-5
            Adam epsilon = 1e-8

        """
        if not args:
            self.mlm_args = word_prediction_args
        else:
            self.mlm_args = args

        # TODO Test the sequence classifier with other models

        if self.model_name != "XLNET":

            # current implementation:
            if not self.mlm:
                self._get_masked_language_model()  # if already has self.mlm
                # don't call this
            self.mwp_trainer = FinetuneMlm(self.mlm, self.mlm_args,
                                           self.tokenizer, self.logger)

            self.logger.info(
                "You can now train a masked word prediction model using %s",
                self.model_name)

        else:
            raise ValueError(
                "Masked language model training is not available for XLNET")

    def train_mwp(self, train_path: str):
        """
        Trains the model with masked language modeling loss.

        train_path: Path to the training file, expected to be a .txt or of
        similar form.

        """

        if torch.cuda.is_available():
            if self.mwp_trained and self.mwp_trainer:  # If model is trained
                self.logger.warning("Training on the already fine-tuned model")
                self.mwp_trainer.train(train_path)

            elif self.mwp_trainer and not self.mwp_trained:  # If trainer
                # exists but isn't trained
                self.mlm, self.tokenizer = self.mwp_trainer.train(train_path)
                self.mwp_trained = True

            elif not self.mwp_trainer:  # If trainer doesn't exist
                raise ValueError(
                    "The model is not loaded, you should run init_train_mwp.")

        else:  # If the user doesn't have a gpu.
            raise ValueError(
                "You are using %s, you must use a GPU to train a MLM",
                self.gpu_support)

    def eval_mwp(self, eval_path: str, batch_size: int = 2):
        """
        Evaluates the masked language model and returns the perplexity and
        the evaluation loss.

        eval_path: Path to the evaluation file, expected to be a .txt or
        similar.
        batch_size: Depending on the gpu the user may increase or decrease
        batch size.

        """
        if not self.mwp_trainer:
            raise ValueError(
                "The model is not loaded, you should run init_train_mwp.")

        if not self.mwp_trained:
            self.logger.warning(
                "You are evaluating on the pretrained model, not the fine-tuned model.")

        results = self.mwp_trainer.evaluate(eval_path, batch_size)

        return results
