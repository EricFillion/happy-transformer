"""
This code is a modified version of the official documentation for the
transformer library by Hugging Face which can be found below.

We prioritized following the official documentation as close as possible to ensure we're using
robust methods. And also, to improve maintainability as they update the documentation.

https://huggingface.co/transformers/custom_datasets.html#question-answering-with-squad-2-0
"""
import time
import csv
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from happytransformer.trainer import Trainer

class QATrainer(Trainer):

    def __init__(self, model, model_type, tokenizer, device, logger):
        super().__init__(model, model_type, tokenizer, device, logger)

    def train(self, input_filepath, args):
        """
        See docstring in HappyQuestionAnswering.train()
        """
        #todo: add time elapsed and test time remaining similar to what is within eval

        contexts, questions, answers = self.__get_data(input_filepath)
        self.__add_end_idx(contexts, answers)
        encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)
        self.__add_token_positions(encodings, answers)
        dataset = QuestionAnsweringDataset(encodings)
        self.model.train()

        train_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

        optim = AdamW(self.model.parameters(), lr=args['learning_rate'])

        for epoch in range(args['epochs']):
            epoch_output = "Epoch: " + str(epoch) + "\n\n"
            self.logger.info(epoch_output)
            batch_num = 0
            for batch in train_loader:
                batch_output = "Batch: " + str(batch_num)
                self.logger.info(batch_output)
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()
                batch_logger_output = "Batch: " + str(batch_num)\
                                      + "   loss: " + str(round(loss.item(), 6))
                self.logger.info(batch_logger_output)
                batch_num += 1
        self.model.eval()

    def eval(self, input_filepath, solve, output_filepath, args):
        """
        See docstring in HappyQuestionAnswering.eval()

        solve: HappyQuestionAnswering.answers_to_question()
        """

        contexts, questions, answers = self.__get_data(input_filepath)
        init_time = time.time()
        correct = 0
        count = 0
        total = len(contexts)
        update_interval = self._get_update_interval(total)

        results = list()

        for case in zip(contexts, questions, answers):
            context = case[0]
            question = case[1]
            answer = case[2]

            result = solve(question, context, k=1000)[0]
            output_text = result["text"]
            output_softmax = result["softmax"]

            # todo modify the qa functionality to output with correct capitalization

            compare_answer = answer["answer_text"].lower()

            if output_text == compare_answer:
                correct += 1

            results.append(
                {
                    "contexts": context,
                    "questions": question,
                    "answer": answer["answer_text"].lower(),
                    "outputs": output_text,
                    "correct": output_text == compare_answer,
                    "softmax": output_softmax

                }
                )
            count += 1

            self._print_status(init_time, count, total, update_interval, correct/count)

        score = correct/total
        ending = str(round(score, 2) * 100) + "%"

        result_output = "Evaluating Result: " + str(correct) + "/" + str(total) + " -- " + ending
        self.logger.info(result_output)

        fieldnames = ["contexts", "questions", "answer", "outputs", "correct", "softmax"]
        self._output_result_to_csv(output_filepath, fieldnames, results)

        return score

    def test(self, input_filepath, solve, output_filepath, args):
        """
        See docstring in HappyQuestionAnswering.test()

        solve: HappyQuestionAnswering.answers_to_question()

        """
        contexts, questions = self.__get_data(input_filepath, test_data=True)
        init_time = time.time()
        total = len(contexts)
        count = 0
        update_interval = self._get_update_interval(total)

        results = list()

        for case in zip(contexts, questions):
            context = case[0]
            question = case[1]

            result = solve(question, context, k=1000)[0]
            output_text = result["text"]
            output_softmax = result["softmax"]

            # todo modify the qa functionality to output with correct capitalization
            results.append(
                {
                    "contexts": context,
                    "questions": question,
                    "outputs": output_text,
                    "softmax": output_softmax
                }
                )

            self._print_status(init_time, count, total, update_interval, None)
            count += 1

        fieldnames = ["contexts", "questions", "outputs", "softmax"]
        self._output_result_to_csv(output_filepath, fieldnames, results)

        result_output = "Output saved to: " + output_filepath

        count += 1
        self.logger.info(result_output)

    @staticmethod
    def __get_data(filepath, test_data=False):
        """
        Used for parsing data for training and evaluating (both contain labels)
        :param filepath: a string that contains the location of the data
        :return:
        """
        contexts = []
        questions = []
        answers = []
        with open(filepath, newline='') as csv_file:
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

    @staticmethod
    def __add_end_idx(contexts, answers):
        for answer, context in zip(answers, contexts):

            gold_text = answer['answer_text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # todo (maybe): strip answer['text'] (remove white space from start and end)
            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx - 1:end_idx - 1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
            elif context[start_idx - 2:end_idx - 2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters
            else:
                print("error: implement skipping training answer")

    def __add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


class QuestionAnsweringDataset(torch.utils.data.Dataset):
    """
    A class used to iterate through the training data.
    It used to create  a torch DataLoader object, so that the training data can be
    iterated through in batches easily.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)