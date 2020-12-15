# disable pylint TODO warning
# pylint: disable=W0511
# pylint: disable=C0301

"""
HappyTransformer is a wrapper over pytorch_transformers to make it
easier to use.
"""

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

        # the following variables are declared in the  child class:
        self.tokenizer = None
        self.cls_token = None
        self.sep_token = None
        self.masked_token = None

        # Child class sets to indicate which model is being used
        self.tag_one_transformers = ['BERT', "ROBERTA", 'XLNET']

        # GPU support
        self.gpu_support = torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")

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
        pass

    def predict_mask(self, text: str, options=None, num_results=1):
        """
        Method to predict what the masked token in the given text string is.
        NOTE: This is the generic version of this predict_mask method. If a
        child class needs a different implementation they should overload this
        method, not create a new method.
        :param text: a string with a masked token within it
        :param options: list of options that the mask token may be [optional]
        :param k: the number of options to output if no output list is given
                  [optional]
        :return: list of dictionaries containing the predicted token(s) and
                 their corresponding softmax values
        NOTE: If no options are given, the returned list will be length 1
        """
        if self.mlm is None:
            self._get_masked_language_model()
            
        if self.gpu_support == "cuda":
            self.mlm.to("cuda")

        if self.model_name in self.tag_one_transformers:
            text = text.replace("<mask>", "[MASK]")
            text = text.replace("<MASK>", "[MASK]")
        else:
            text = text.replace("[MASK]", "<mask>")

        self._text_verification(text)

        tokenized_text = self. \
            _get_tokenized_text(text)
        masked_index = tokenized_text.index(self.masked_token)

        softmax = self._get_prediction_softmax(tokenized_text)

        if options is not None:

            if self.model_name == "BERT":
                option_ids = [self.tokenizer.encode(option) for option in options]

                option_ids = option_ids[:num_results]

                scores = list(map(lambda x: self.soft_sum(x, softmax[0],
                                                          masked_index),
                                  option_ids))
                tupled_predictions = tuple(zip(options, scores))

            else:
                top_predictions = torch.topk(softmax[0, masked_index], 5000)
                scores = top_predictions[0].tolist()
                lowest_score = min(float(i) for i in scores)
                prediction_index = top_predictions[1].tolist()
                top_options = self.tokenizer.convert_ids_to_tokens(prediction_index)

                if self.model_name == "XLNET":
                    top_options = self.__remove_starting_character(top_options, "▁")
                if self.model_name == "ROBERTA":
                    top_options = self.__remove_starting_character(top_options, "Ġ")
                    top_options = self.__switch_prediction(top_options, "</s>", '.')

                option_scores = list()
                for option in options:
                    if option in top_options:
                        option_id = top_options.index(option)
                        option_scores.append(scores[option_id])
                    else:
                        option_scores.append(lowest_score)

                tupled_predictions = tuple(zip(options, option_scores))

                sorted(tupled_predictions, key=lambda x: x[1])

                tupled_predictions = tupled_predictions[:num_results]


        else:
            top_predictions = torch.topk(softmax[0, masked_index], num_results)
            scores = top_predictions[0].tolist()
            prediction_index = top_predictions[1].tolist()
            options = self.tokenizer.convert_ids_to_tokens(prediction_index)

            if self.model_name == "XLNET":  # TODO find other models that also require this
                options = self.__remove_starting_character(options, "▁")
            if self.model_name == "ROBERTA":
                options = self.__remove_starting_character(options, "Ġ")
                options = self.__switch_prediction(options, "</s>", '.')
            tupled_predictions = tuple(zip(options, scores))

        if self.gpu_support == "cuda":
            torch.cuda.empty_cache()

        return self.__format_option_scores(tupled_predictions)

    def __switch_prediction(self, options, current_token, new_token):
        """
        Switches a token with a different token in final predictions  for predict_mask.
        So far it is only used to switch the "</s>" token with "." for RoBERTA. "</s>" is meant to indicate
        a new sentence.
        """

        for n, i in enumerate(options):
            if i == current_token:
                options[n] = new_token

        return options

    def __remove_starting_character(self, options, starting_char):
        """
        Some cased models like XLNet place a "▁" character in front of lower cased predictions.
        For most applications this extra bit of information is irrelevant.
        :param options: A list that contains word predictions
        ;param staring_char: The special character that is placed at the start of the predicted word
        :return: a new list of tuples where the prediction's name does not contains a special starting character
        """
        new_predictions = list()
        for prediction in options:
            if prediction[0] == starting_char:
                new_prediction = prediction[1:]
                new_predictions.append(new_prediction)
            else:
                new_predictions.append(prediction)
        return new_predictions

    def _get_tokenized_text(self, text):
        """
        Formats a sentence so that it can be tokenized by a transformer.
        :param text: a 1-2 sentence text that contains [MASK]
        :return: A string with the same sentence that contains the required
                 tokens for the transformer
        """

        # Create a spacing around each punctuation character. eg "!" -> " ! "
        # TODO: easy: find a cleaner way to do punctuation spacing
        text = re.sub('([.,!?()])', r' \1 ', text)
        # text = re.sub('\s{2,}', ' ', text)

        split_text = text.split()
        new_text = list()
        new_text.append(self.cls_token)

        for i, char in enumerate(split_text):
            new_text.append(char.lower())
            if char not in string.punctuation:
                pass
            # must be a punctuation symbol
            elif i + 1 >= len(split_text):
                # is the last punctuation so simply add to the new_text
                pass
            else:
                if split_text[i + 1] in string.punctuation:
                    pass
                else:
                    new_text.append(self.sep_token)
                    # if self.model_name == "ROBERTA":
                    #     # ROBERTA requires two "</s>" tokens to separate sentences
                    #     new_text.append(self.sep_token)
                # must be a middle punctuation
        new_text.append(self.sep_token)

        text = " ".join(new_text).replace('[mask]', self.masked_token)
        text = self.tokenizer.tokenize(text)
        return text

    def _get_prediction_softmax(self, text):
        """
        Gets the softmaxes of the predictions for each index in the the given
        input string.
        Returned tensor will be in shape:
            [1, <tokens in string>, <possible options for token>]
        :param text: a tokenized string to be used by the transformer.
        :return: a tensor of the softmaxes of the predictions of the
                 transformer

        """

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        if self.gpu_support == "cuda":
            tokens_tensor = tokens_tensor.to('cuda')

        with torch.no_grad():

            if self.model_name != "ROBERTA":
                segments_ids = self._get_segment_ids(text)
                segments_tensors = torch.tensor([segments_ids])
                if self.gpu_support == "cuda":
                    segments_tensors = segments_tensors.to('cuda')
                outputs = self.mlm(tokens_tensor, token_type_ids=segments_tensors)
            else:
                outputs = self.mlm(tokens_tensor)

            predictions = outputs[0]

            softmax = self._softmax(predictions)
            return softmax

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
        ranked_scores = sorted(tupled_predicitons, key=lambda x: x[1],
                               reverse=True)
        formatted_ranked_scores = list()
        for word, softmax in ranked_scores:
            formatted_ranked_scores.append({'word': word, 'softmax': softmax})
        return formatted_ranked_scores

    def _softmax(self, value):
        # TODO: make it an external function
        return value.exp() / (value.exp().sum(-1)).unsqueeze(-1)

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
        split_location = tokenized_text.index(self.sep_token)
        segment_ids = list()
        for i in range(0, len(tokenized_text)):
            if i <= split_location:
                segment_ids.append(0)
            else:
                segment_ids.append(1)
            # add exception case for XLNet

        return segment_ids

    def _text_verification(self, text: str):

        # TODO,  Add cases for the other masked tokens used in common transformer models
        valid = True
        if '[MASK]' not in text:
            self.logger.error("[MASK] was not found in your string. Change the word you want to predict to [MASK]")
            valid = False
        if '<mask>' in text or '<MASK>' in text:
            self.logger.info('Instead of using <mask> or <MASK>, use [MASK] please as it is the convention')
            valid = True
        if '[CLS]' in text:
            self.logger.error("[CLS] was found in your string.  Remove it as it will be automatically added later")
            valid = False
        if '[SEP]' in text:
            self.logger.error("[SEP] was found in your string.  Remove it as it will be automatically added later")
            valid = False
        if not valid:
            exit()

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
            self.logger.error("Initialize the sequence classifier before training")
            exit()

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
            self.logger.error("Train the sequence classifier before evaluation")
            exit()

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

        # todo finish
        if not self.seq_trained:
            self.logger.error("Train the sequence classifier before testing")
            exit()

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
            self.logger.error(
                "Masked language model training is not available for XLNET")
            sys.exit()

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
                self.logger.error(
                    "The model is not loaded, you should run init_train_mwp.")
                sys.exit()

        else:  # If the user doesn't have a gpu.
            self.logger.error(
                "You are using %s, you must use a GPU to train a MLM",
                self.gpu_support)
            sys.exit()

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
            self.logger.error(
                "The model is not loaded, you should run init_train_mwp.")
            sys.exit()

        if not self.mwp_trained:
            self.logger.warning(
                "You are evaluating on the pretrained model, not the fine-tuned model.")

        results = self.mwp_trainer.evaluate(eval_path, batch_size)

        return results
