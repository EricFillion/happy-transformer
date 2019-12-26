# disable pylint TODO warning
# pylint: disable=W0511


"""
HappyTransformer is a wrapper over pytorch_transformers to make it
easier to use.

"""
import string
import re
import numpy as np
import torch


class HappyTransformer:
    """
    Initializes pytroch's transformer models and provided methods for
    their basic functionality.

    Philosophy: Automatically make decisions for the user so that they don't
                have to have any understanding of PyTorch or transformer
                models to be able to utilize their capabilities.
    """

    def __init__(self, model):
        # Transformer and tokenizer set in child class
        self.mlm = None  # Masked Language Model
        self.nsp = None  # Next Sentence Prediction
        self.qa = None   # Question Answering
        self.model_to_use = model
        self.tokenizer = None
        # Child class sets to indicate which model is being used
        self.model = ''
        self.tag_one_transformers = ['BERT', 'GPT2', 'XLM', 'XLNET']

        # GPU support
        self.gpu_support = torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")
        print("Using model:", self.gpu_support)

    def _get_initial_transformers(self, initial_transformers):
        for transformer in initial_transformers:
            if transformer == 'mlm':
                self._get_masked_language_model()
            if transformer == 'nsp':
                self._get_next_sentence_prediction()
            if transformer == 'qa':
                self._get_question_answering()

    def _get_masked_language_model(self):
        # Must be overloaded
        # TODO make an exception to be thrown if not overloaded
        pass

    def _get_next_sentence_prediction(self):
        # Must be overloaded
        # TODO make an exception to be thrown if not overloaded
        pass

    def _get_question_answering(self):
        # Must be overloaded
        # TODO make an exception to be thrown if not overloaded
        pass

    def predict_mask(self, text: str, options=None, k=1):
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
                 their corresponding score(s).

        NOTE: If no options are given, the returned list will be length 1
        """
        if self.mlm is None:
            self._get_masked_language_model()

        if self.model in self.tag_one_transformers:
            text = text.replace("<mask>", "[MASK]")
            text = text.replace("<MASK>", "[MASK]")
        else:
            text = text.replace("[MASK]", "<mask>")

        if not self._text_verification(text):
            return

        tokenized_text = self.__get_tokenized_text(text)
        masked_index = tokenized_text.index(self.masked_token)
        softmax = self._get_prediction_softmax(tokenized_text)
        if options is not None:
            option_ids = [self.tokenizer.encode(option) for option in options]

            scores = list(map(lambda x: self.soft_sum(x, softmax[0],
                                                      masked_index),
                              option_ids))
        else:
            top_predictions = torch.topk(softmax[0, masked_index], k)
            scores = top_predictions[0].tolist()
            prediction_index = top_predictions[1].tolist()
            options = self.tokenizer.convert_ids_to_tokens(prediction_index)

        tupled_predictions = tuple(zip(options, scores))

        if self.model == "XLNET": # TODO find other models that also require this
            tupled_predictions = self.__remove_staring_character(tupled_predictions, "▁")
        if self.model == "RoBERTa":
            tupled_predictions = self.__remove_staring_character(tupled_predictions, "Ġ")


        if self.gpu_support == "cuda":
            torch.cuda.empty_cache()

        return self.__format_option_scores(tupled_predictions)

    def __remove_staring_character(self, tupled_predictions, starting_char):
        """
        Some cased models like XLNet place a "▁" character in front of lower cased predictions.
        For most applications this extra bit of information is irrelevant.

        :param tupled_predictions: A list that contains tuples where the first index is
                                the name of the prediction and the second index is the
                                prediction's softmax
        ;param staring_char: The special character that is placed at the start of the predicted word
        :return: a new list of tuples where the prediction's name does not contains a special starting character
        """
        new_predictions = list()
        for prediction in tupled_predictions:
            word_prediction = prediction[0]
            if word_prediction[0] == starting_char:
                new_prediction = (word_prediction[1:], prediction[1])
                new_predictions.append(new_prediction)
            else:
                new_predictions.append(prediction)
        return new_predictions

    def __get_tokenized_text(self, text):
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
            elif i+1 >= len(split_text):
                # is the last punctuation so simply add to the new_text
                pass
            else:
                if split_text[i + 1] in string.punctuation:
                    pass
                else:
                    new_text.append(self.sep_token)
                # must be a middle punctuation
        new_text.append(self.sep_token)
        text = " ".join(new_text).replace('[mask]', self.masked_token)
        text = self.tokenizer.tokenize(text)
        return text

    def _get_prediction_softmax(self, text: str):
        """
        Gets the softmaxes of the predictions for each index in the the given
        input string.
        Returned tensor will be in shape:
            [1, <tokens in string>, <possible options for token>]

        :param text: a tokenized string to be used by the transformer.
        :return: a tensor of the softmaxes of the predictions of the
                 transformer
        """
        segments_ids = self._get_segment_ids(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.mlm(tokens_tensor,
                                       token_type_ids=segments_tensors)
            predictions = outputs[0]

            softmax = self.__softmax(predictions)
            return softmax

    def __format_option_scores(self, tupled_predicitons: list):
        """
        Formats the given list of tuples containing the option and its
        corresponding score into a user friendly list of dictionaries where
        the first element in the list is the option with the highest score.
        Dictionary will be in the form:
             {'word': <the option>, 'score': <score for the option>}

        :param: ranked_scores: list of tuples to be converted into user
                friendly dicitonary
        :return: formatted_ranked_scores: list of dictionaries of the ranked
                 scores
        """
        ranked_scores = sorted(tupled_predicitons, key=lambda x: x[1],
                               reverse=True)
        formatted_ranked_scores = list()
        for word, score in ranked_scores:
            formatted_ranked_scores.append({'word': word, 'score': score})
        return formatted_ranked_scores

    def __softmax(self, value):
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

    def finish_sentence(self, text: str, maxPredictionLength=100):
        """

        :param text: a string that is the start of a sentence to be finished
        :param maxPredictionLength: an int with the maximum number of words to
                                    be predicted
        :return: the completed sentence
        """
        if self.mlm is None:
            self._get_masked_language_model()

        father_predict = ""
        grand_father_predict = ""

        for i in range(0, maxPredictionLength):
            predict_text = text + self.masked_token
            predict_word = self.predict_mask(predict_text)[0]

            if predict_word == father_predict\
                    and predict_word == grand_father_predict:
                # if the same token was predicted three times in a row
                return text

            grand_father_predict = father_predict
            father_predict = predict_word

            text = text + predict_word
        return text

    def _text_verification(self, text: str):

        # TODO,  Add cases for the other masked tokens used in common transformer models
        valid = True
        if '[MASK]' not in text:
            print("[MASK] was not found in your string. Change the word you want to predict to [MASK]")
            valid = False
        if '<mask>' in text or '<MASK>' in text:
            print('Instead of using <mask> or <MASK>, use [MASK] please as it is the convention')
            valid = True 
        if '[CLS]' in text:
            print("[CLS] was found in your string.  Remove it as it will be automatically added later")
            valid = False
        if '[SEP]' in text:
            print("[SEP] was found in your string.  Remove it as it will be automatically added later")
            valid = False

        return valid

    def is_next_sentence(self, a, b):
        """
        Determines if sentence B is likely to be a continuation after sentence
        A.

        :param a: First sentence
        :param b: Second sentence to test if it comes after the first
        :return tuple: True if b is likely to follow a, False if b is unlikely
                       to follow a, with the probabilities as the second item
                       of the tuple
        """
        if self.nsp is None:
            self._get_next_sentence_prediction()
        connected = a + ' ' + b
        tokenized_text = self.__get_tokenized_text(connected)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = self._get_segment_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            predictions = self.nsp(tokens_tensor, token_type_ids=segments_tensors)[0]
        print(type(predictions))
        softmax = self.__softmax(predictions)
        if torch.argmax(softmax) == 0:
            return (True, softmax)
        else:
            return (False, softmax)

    def answer_question(self, question, context):
        """
        Using the given context returns the answer to the given question.

        :param question: 
        :param context:
        """
        if self.qa is None:
            self._get_question_answering()
        input_text = "[CLS] " + question + " [SEP] " + context + " [SEP]"
        input_ids = self.tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1
                          for i in range(len(input_ids))]
        token_tensor = torch.tensor([input_ids])
        segment_tensor = torch.tensor([token_type_ids])
        with torch.no_grad():
            start_scores, end_scores = self.qa(token_tensor,
                                               token_type_ids=segment_tensor)
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_list = all_tokens[torch.argmax(start_scores):
                                 torch.argmax(end_scores)+1]
        answer = self.tokenizer.convert_tokens_to_string(answer_list)
        answer = answer.replace(' \' ', '\' ').replace('\' s ', '\'s ')
        return answer

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
