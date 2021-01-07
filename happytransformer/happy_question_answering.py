"""
Contains the HappyQuestionAnswering class.

"""

import torch
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.qa.util import qa_probabilities
from happytransformer.qa.trainer import QATrainer
from happytransformer.qa.default_args \
    import ARGS_QA_EVAL, ARGS_QA_TEST, ARGS_QA_TRAIN
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast

)


class HappyQuestionAnswering(HappyTransformer):
    """
    This class is a user facing class that allows users to solve question answering problems using
    a transformer QA models. These models are able to answer a question given context for the
    question by selecting a span within the context that answers the question.

    The purpose of this class is to be lightweight and easy
    to understand and to offload complex tasks to
    other classes.
    """
    def __init__(self, model_type="BERT",
                 model_name="bert-large-uncased-whole-word-masking-finetuned-squad", device=None):
        model = BertForQuestionAnswering.from_pretrained(model_name)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

        super().__init__(model_type, model_name, model, tokenizer, device)

        self._trainer = QATrainer(model, model_type, tokenizer, self._device,
                                  self.logger)

    def answers_to_question(self, question, context, k=3):
        input_ids = self._tokenize_qa(question, context)
        qa_output = self._run_qa_model(input_ids)
        sep_id_index = input_ids.index(self._tokenizer.sep_token_id)
        probabilities = qa_probabilities(
            # only consider logits from the context part of the embedding.
            # that is, between the middle [SEP] token
            # and the final [SEP] token
            start_logits=qa_output.start_logits[0][sep_id_index+1:-1],
            end_logits=qa_output.end_logits[0][sep_id_index+1:-1],
            k=k
        )
        # qa probabilities use indices relative to context.
        # tokens use indices relative to overall question [SEP] context embedding.
        # need offset to resolve this difference
        token_offset = sep_id_index + 1
        return [
            # grab ids from start to end (inclusive) and decode to text
            {"text": self._tokenizer.decode(
                input_ids[token_offset+answer.start_idx: token_offset+answer.end_idx+1]
                ),
             "softmax": answer.probability}

            for answer in probabilities
        ]

    def answer_question(self, question, text):
        """
        Using the given text, find the answer to the given question and return it.

        :param question: The question to be answered
        :param text: The text containing the answer to the question
        :return: The answer to the given question, as a string
        """
        return self.answers_to_question(question, text, 1)[0]["text"]

    def _run_qa_model(self, input_ids):
        sep_id_index = input_ids.index(self._tokenizer.sep_token_id)
        before_after_ids = [
            0 if idx <= sep_id_index else 1
            for idx, _ in enumerate(input_ids)
        ]
        with torch.no_grad():
            return self._model(
                input_ids=torch.tensor([input_ids]),
                token_type_ids=torch.tensor([before_after_ids])
            )

    def _tokenize_qa(self, question, context):
        input_text = ' '.join([
            question,
            self._tokenizer.sep_token,
            context
        ])
        input_ids = self._tokenizer.encode(input_text)
        return input_ids

    def train(self, input_filepath, args=ARGS_QA_TRAIN):
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values: context,
         question, answer_text, answer_start
        args: a dictionary that contains settings found under
        happytransformer.happytasks.happy_qa.default_args.py
        return: None
        """
        self._trainer.train(input_filepath=input_filepath, args=args)

    def eval(self, input_filepath, output_filepath, args=ARGS_QA_EVAL):
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
         context, question, answer_text, answer_start
        args: a dictionary that contains settings found under
        happytransformer.happytasks.happy_qa.default_args.py
        output_filepath: a path to a csv file to output the results.
        This file contains the following header values: contexts,
        questions, answer, outputs, correct, softmax
        return: correct ration (correct/total)
        """
        return self._trainer.eval(input_filepath=input_filepath,
                                  solve=self.answers_to_question, args=args,
                                  output_filepath=output_filepath)

    def test(self, input_filepath, output_filepath, args=ARGS_QA_TEST):
        """
        Tests the question answering model. Used to obtain results

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
         context, question
        args: a dictionary that contains settings found under
        happytransformer.happytasks.happy_qa.default_args.py
        output_filepath: a path to a csv file to output the results.
        This file contains the following header values: contexts, questions, outputs, softmax
        return: None
        """
        self._trainer.test(input_filepath=input_filepath,
                           solve=self.answers_to_question, args=args,
                           output_filepath=output_filepath)
