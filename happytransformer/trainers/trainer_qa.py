"""
This code is a modified version of the official documentation for the transformer library by Hugging Face
which can be found below.

We prioritized following the official documentation as close as possible to ensure we're using
robust methods. And also, to improve maintainability as they update the documentation.

https://huggingface.co/transformers/custom_datasets.html#question-answering-with-squad-2-0
"""

import torch
from happytransformer.trainer import Trainer
import csv
from torch.utils.data import DataLoader
from transformers import AdamW
import time
from happytransformer.trainers.default_args.default_args_qa import ARGS_QA_TRAINING

class QATrainer(Trainer):

    def __init__(self, model, model_type, tokenizer, device, runner, logger):
        super(QATrainer, self).__init__(model, model_type, tokenizer, device, runner, logger)

    def train(self, filepath, args=ARGS_QA_TRAINING):
        """
        #todo: add time elapsed and test time remaining similar to what is within eval


        :param filepath:
        :param args:
        :return:
        """

        if args == None:
            args = ARGS_QA_TRAINING

        contexts, questions, answers = self.__get_train_eval_data(filepath)

        self.__add_end_idx(contexts, answers)
        encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)
        self.__add_token_positions(encodings, answers)
        dataset = QuestionAnsweringDataset(encodings)
        self.model.train()

        train_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

        optim = AdamW(self.model.parameters(), lr=args['learning_rate'])

        for epoch in range(args['epochs']):
            epoch_output = "Epoch: " + str(epoch)
            self.logger.info(epoch_output)
            batch_num = 0
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                     end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()
                batch_logger_output = "Batch: " + str(batch_num) + "   loss: " + str(round(loss.item(), 6))
                self.logger.info(batch_logger_output)
                batch_num += 1
        self.model.eval()

    def eval(self, filepath, args, output_filepath=None):

        contexts, questions, answers = self.__get_train_eval_data(filepath)
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
            output = self.runner.answer_question(question, context)

            # todo modify the qa functionality to output with correct capitalization

            compare_answer = answer["answer_text"].lower()

            if output == compare_answer:
                correct += 1
            count += 1

            if output_filepath != None:
                results.append(
                    {
                        "contexts": context,
                        "questions": question,
                        "answer": answer["answer_text"].lower(),
                        "outputs": output,
                        "correct": output == compare_answer
                    }
                )

            self._print_status(init_time, count, total, update_interval, correct/count)

        score = correct/total
        ending = str(round(score, 2) * 100) + "%"

        result_output = "Evaluating Result: " + str(correct) + "/" + str(total) + " -- " + ending
        self.logger.info(result_output)

        if output_filepath != None:
            fieldnames = ["contexts", "questions", "answer", "outputs", "correct"]
            self._output_result_to_csv(output_filepath, fieldnames, results)

        return score


    def test(self, filepath, args, output_filepath):
        #todo
        pass

    @staticmethod
    def __get_train_eval_data(filepath):
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
                answer = {}
                answer["answer_text"] = row['answer_text']
                answer["answer_start"] = int(row['answer_start'])
                answers.append(answer)
        csv_file.close()

        return contexts, questions, answers

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

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)