"""
Contains the HappyQuestionAnswering class.

"""
from typing import List
from dataclasses import dataclass
from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, DataCollatorWithPadding
from happytransformer.happy_transformer import HappyTransformer
from happytransformer.args import QATrainArgs, QAEvalArgs, QATestArgs
from happytransformer.happy_trainer import EvalResult
import csv
from tqdm import tqdm
from happytransformer.adaptors import get_adaptor

@dataclass
class QuestionAnsweringResult:
    answer: str
    score: float
    start: int
    end: int


class HappyQuestionAnswering(HappyTransformer):
    """
    This class is a user facing class that allows users to solve question answering problems using
    a transformer QA models. These models are able to answer a question given context for the
    question by selecting a span within the context that answers the question.

    The purpose of this class is to be lightweight and easy
    to understand and to offload complex tasks to
    other classes.
    """
    def __init__(self, model_type="DISTILBERT",
                 model_name="distilbert-base-cased-distilled-squad", 
                 load_path: str = "",
                 use_auth_token: str = None, from_tf=False):
        
        self.adaptor = get_adaptor(model_type)

        if load_path != "":
            model = AutoModelForQuestionAnswering.from_pretrained(load_path, from_tf=from_tf)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, use_auth_token=use_auth_token, from_tf=from_tf)

        super().__init__(model_type, model_name, model, use_auth_token=use_auth_token, load_path=load_path)

        self._pipeline = QuestionAnsweringPipeline(model=self.model, tokenizer=self.tokenizer, device=self.device)


        self._data_collator = DataCollatorWithPadding(self.tokenizer)
        self._t_data_file_type = "csv"

        self._type = "qa"

    def answer_question(self, context: str, question: str, top_k: int = 1) \
            -> List[QuestionAnsweringResult]:
        """
        Find the answers to a question.
        The answer MUST be contained somewhere within the context for this to work.
        top_k describes the number of answers to return.
        """

        pipeline_output = self._pipeline(context=context, question=question, topk=top_k)
        # transformers returns a single dictionary when top_k ==1.
        # Our convention however is to have constant output format
        answers = [pipeline_output] if top_k == 1 else pipeline_output

        return [
            QuestionAnsweringResult(
                answer=answer["answer"],
                score=answer["score"],
                start=answer["start"],
                end=answer["end"],)
            for answer in answers
        ]

    def train(self, input_filepath, eval_filepath: str= "",  args: QATrainArgs =QATrainArgs()):
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values: context,
        question, answer_text, answer_start

        args: Either a QATrainArgs() object or a dictionary that contains all of the same keys as ARGS_QA_TRAIN

        return: None
        """
        super(HappyQuestionAnswering, self).train(input_filepath, args, eval_filepath)

    def eval(self, input_filepath, args=QAEvalArgs()) -> EvalResult:
        """
        Trains the question answering model

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        context, question, answer_text, answer_start

        args: Either a QAEvalArgs() object or a dictionary that contains all of the same keys as ARGS_QA_EVAl

        return: A dictionary that contains a key called "eval_loss"

        """
        return super(HappyQuestionAnswering, self).eval(input_filepath, args)


    def test(self, input_filepath, args=QATestArgs()):
        """
        Tests the question answering model. Used to obtain results

        input_filepath: a string that contains the location of a csv file
        for training. Contains the following header values:
        context, question

        args: Either a QATestArgs() object or a dictionary that contains all of the same keys as ARGS_QA_TEST

        return: A list of dictionaries. Each dictionary
        contains the keys: "score", "start", "end" and "answer"
        """
        if type(args) == dict:
            raise ValueError( "As of version 2.5.0 dictionary inputs are not acceptable. Please provide a QATestArgs. ")


        if args.save_preprocessed_data:
            self.logger.info("Saving preprocessed data is currently "
                             "not available for question answering models. "
                             "It will be added soon. ")
        if args.load_preprocessed_data:
            self.logger.info("Loading preprocessed data is currently "
                             "not available for question answering models. "
                             "It will be added soon. ")

        contexts, questions = self._get_data(input_filepath, test_data=True)

        return [
            self.answer_question(context, question)[0]
            for context, question in
            tqdm(zip(contexts, questions))
        ]

    def _tok_function(self, raw_dataset, dataclass_args: QATrainArgs):

        def __preprocess_function(case):
            print(case)
            case["answer_start"] = int(case['answer_start'])
            gold_text = case['answer_text']
            start_idx = case['answer_start']
            end_idx = start_idx + len(gold_text)

            # todo (maybe): strip answer['text'] (remove white space from start and end)
            # sometimes squad answers are off by a character or two – fix this
            if case["context"][start_idx:end_idx] == gold_text:
                case['answer_end'] = end_idx
            elif case["context"][start_idx - 1:end_idx - 1] == gold_text:
                case["context"]['answer_start'] = start_idx - 1
                case["context"]['answer_end'] = end_idx - 1
            elif case[start_idx - 2:end_idx - 2] == gold_text:
                case["context"]['answer_start'] = start_idx - 2
                case["context"]['answer_end'] = end_idx - 2

            encodings = self.tokenizer(case["context"], case["question"], truncation=True, padding=True)

            start_position = encodings.char_to_token(case['answer_start'])
            end_position =  encodings.char_to_token(case['answer_end'] - 1)
            if start_position is None:
                start_position = self.tokenizer.model_max_length
            if end_position is None:
                end_position = self.tokenizer.model_max_length

            encodings.update({'start_positions': start_position, 'end_positions': end_position})

            return encodings


        tok_dataset = raw_dataset.map(
            __preprocess_function,
            batched=False,
            num_proc=1,
            remove_columns=["context", "question", "answer_text", "answer_start"],
            desc="Tokenizing data"
        )

        return tok_dataset
    @staticmethod
    def _get_data(filepath, test_data=False):
        """
        Used to collect
        :param filepath: a string that contains the location of the data
        :return: if test_data = False contexts, questions, answers (all strings)
        else: contexts, questions
        """
        contexts = []
        questions = []
        answers = []
        with open(filepath, newline='', encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                contexts.append(row['context'])
                questions.append(row['question'])
                if not test_data:
                    answer = {}
                    answer["answer_text"] = row['answer_text']
                    answer["answer_start"] = int(row['answer_start'])
                    answers.append(answer)
        csv_file.close()

        if not test_data:
            return contexts, questions, answers
        return contexts, questions
