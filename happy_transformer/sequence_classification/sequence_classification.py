"""
Binary Sequence Classifier for BERT, XLNET and RoBERTa that has fine tuning capabilities.

"""

# pylint: disable=C0301

from __future__ import absolute_import, division, print_function
import math
import numpy as np
from tqdm import tqdm_notebook, trange
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (BertForSequenceClassification,
                                  XLNetForSequenceClassification,
                                  RobertaForSequenceClassification)

from pytorch_transformers import AdamW, WarmupLinearSchedule


from happy_transformer.sequence_classification.classifier_utils import convert_examples_to_features, \
                                               output_modes, \
                                               processors

class SequenceClassifier():
    """
    Sequence Classifier with fine tuning capabilities
    """

    def __init__(self, args, tokenizer, logger):
        self.args = args
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None
        self.model_classes = {
            'BERT': (BertForSequenceClassification),
            'XLNET': (XLNetForSequenceClassification),
            'ROBERTA': (RobertaForSequenceClassification)
        }
        self.train_list_data = None
        self.eval_list_data = None
        self.test_list_data = None
        self.features = False
        self.features_exists = False
        self.tokenizer = tokenizer
        self.logger = logger


        self.model_class = self.model_classes[self.args['model_type']]

        self.model = self.model_class.from_pretrained(self.args['model_name'])
        self.model.to(self.args['gpu_support'])



    def check_task(self):
        "Checks to make sure the task is valid. Currently only \"Binary\" is accepted"
        task = self.args['task_mode']

        if task in processors.keys() and task in output_modes.keys():
            self.processor = processors[task]()
        else:
            raise KeyError(f'{task} is not available')


    def train_model(self):
        """
        Does the proper checks and initializations before training self.model. Then, saves the model
        :return:
        """
        self.check_task()

        self.train_dataset = self.__load_and_cache_examples()
        self.__train()

        # Takes care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        self.model = model_to_save # new


    def __train(self):
        """
        Trains the binary sequence classifier
        """
        sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=sampler,
                                      batch_size=self.args['train_batch_size'])

        t_total = len(train_dataloader) \
            // self.args['gradient_accumulation_steps'] * \
            self.args['num_train_epochs']

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args['weight_decay']},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(t_total * self.args['warmup_ratio'])
        self.args['warmup_steps'] = warmup_steps if self.args['warmup_steps'] == 0 else self.args['warmup_steps']

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['learning_rate'], eps=self.args['adam_epsilon'])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args['warmup_steps'], t_total=t_total)

        if self.args['fp16']:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args['fp16_opt_level'])


        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(int(self.args['num_train_epochs']), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args['gpu_support']) for t in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                # print("\r%f" % loss, end='')

                if self.args['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args['gradient_accumulation_steps']

                if self.args['fp16']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args['max_grad_norm'])

                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % self.args['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    

    def __get_eval_report(self, labels, preds):
        """
        :param labels: Correct answers
        :param preds: predictions
        :return: a confusion matrix
        """
        assert len(preds) == len(labels)

        true_negative, false_positive, false_negative, true_positive = confusion_matrix(labels, preds).ravel()
        return {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative
            }

    def evaluate(self):
        """
        Evaluates the model against a set of questions to determine accuracy
        :return: a dictionary confusion martrix
        """
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        self.check_task()

        self.eval_dataset = self.__load_and_cache_examples()

        results = {}

        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.args['gpu_support']) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)


        preds = np.argmax(preds, axis=1)


        result = self.__get_eval_report(out_label_ids, preds)
        results.update(result)

        return results

    def test(self):
        """
        Generates answers for an input

        :return: a list of answers where each index contains the answer 1 or 0
                for the corresponding test question with the same index
        """
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        self.check_task()

        self.eval_dataset = self.__load_and_cache_examples()

        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.args['gpu_support']) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)


        return preds.tolist()

    def __load_and_cache_examples(self):
        """
        Converts the proper list_data variable to a TensorDataset for the current task
        :return: a TensorDataset for the requested task
        """
        self.processor = processors[self.args["task_mode"]]()
        output_mode = "classification"

        if not self.features_exists and not self.args['reprocess_input_data']:
            self.logger.info("Loading features from cached file %s")

        else:
            self.features_exists = True
            label_list = self.processor.get_labels()

            if self.args['task'] == 'eval':
                examples = self.processor.get_dev_examples(self.eval_list_data)
            elif self.args['task'] == 'train':
                examples = self.processor.get_train_examples(self.train_list_data)
            else:
                examples = self.processor.get_dev_examples(self.test_list_data)

            self.features = convert_examples_to_features(examples, label_list, self.args['max_seq_length'], self.tokenizer,
                                                         output_mode,
                                                         cls_token_at_end=bool(self.args['model_type'] in ['XLNET']),
                                                         # xlnet has a cls token at the end
                                                         cls_token=self.tokenizer.cls_token,
                                                         cls_token_segment_id=2 if self.args['model_type'] in [
                                                             'XLNET'] else 0,
                                                         sep_token=self.tokenizer.sep_token,
                                                         sep_token_extra=bool(self.args['model_type'] in ['ROBERTA']),
                                                         # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                         pad_on_left=bool(self.args['model_type'] in ['XLNET']),
                                                         # pad on the left for xlnet
                                                         pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                         pad_token_segment_id=4 if self.args['model_type'] in [
                                                             'XLNET'] else 0)

        all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in self.features], dtype=torch.long)


        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
