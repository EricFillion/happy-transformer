from __future__ import absolute_import, division, print_function
import glob
import logging
import os
import math
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm_notebook, trange

from tensorboardX import SummaryWriter


from pytorch_transformers import (BertForSequenceClassification,
                                XLMForSequenceClassification,
                                XLNetForSequenceClassification,
                                RobertaForSequenceClassification)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from sklearn.metrics import confusion_matrix

from happy_transformer.classifier_utils import convert_examples_to_features, output_modes, processors


class SequenceClassifier():

    def __init__(self, args, tokenizer):
        self.args = args
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None
        self.model_classes = {
            'bert': (BertForSequenceClassification),
            'xlnet': (XLNetForSequenceClassification),
            'xlm': (XLMForSequenceClassification),
            'roberta': (RobertaForSequenceClassification)
        }
        self.train_list_data = None
        self.eval_list_data = None
        self.features = False
        self.features_exists = False
        self.tokenizer = tokenizer

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model_class = self.model_classes[self.args['model_type']]
        self.model = self.model_class.from_pretrained(self.args['model_name'])
        self.model.to(self.args['gpu_support'])


    def run_sequence_classifier(self):

        task = self.args['task_name']

        if task in processors.keys() and task in output_modes.keys():
            self.processor = processors[task]()
        else:
            raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

        if self.args['do_train']:
            self.train_model()

        if self.args['do_eval']:
            self.eval_dataset = self.load_and_cache_examples(task="binary", evaluate=True, tokenizer=self.tokenizer)

            result = self.evaluate()

            return result


    def train_model(self):
        self.train_dataset = self.load_and_cache_examples(task="binary", tokenizer=self.tokenizer, evaluate=False)
        self.train()

        model_to_save = self.model.module if hasattr(self.model,'module') else self.model  # Takes care of distributed/parallel training

        self.model = model_to_save # new


    def train(self):

        tb_writer = SummaryWriter()

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args['train_batch_size'])

        t_total = len(train_dataloader) // self.args['gradient_accumulation_steps'] * self.args['num_train_epochs']

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

        self.logger.info("***** Running training *****")

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
                          'token_type_ids': batch[2] if self.args['model_type'] in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                print("\r%f" % loss, end='')

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

                    if self.args['logging_steps'] > 0 and global_step % self.args['logging_steps'] == 0:
                        # Log metrics
                        if self.args[
                            'evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate()
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / self.args['logging_steps'], global_step)
                        logging_loss = tr_loss



    def get_eval_report(self, labels, preds):
        assert len(preds) == len(labels)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        return {
                   "tp": tp,
                   "tn": tn,
                   "fp": fp,
                   "fn": fn
               }

    def evaluate(self):
        # Loop to handle MNLI double evaluation (matched, mis-matched)

        results = {}

        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

        # Eval!
        self.logger.info("***** Running evaluation *****")
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
                          'token_type_ids': batch[2] if self.args['model_type'] in ['bert', 'xlnet'] else None,
                          # XLM does not  use segment_ids
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

        if self.args['output_mode'] == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.args['output_mode'] == "regression":
            preds = np.squeeze(preds)

        print("out_label_ids", out_label_ids)
        result = self.get_eval_report(out_label_ids, preds)
        results.update(result)

        return results

    def test(self):
        # Loop to handle MNLI double evaluation (matched, mis-matched)

        results = {}

        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

        # Eval!
        self.logger.info("***** Running evaluation *****")
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
                          'token_type_ids': batch[2] if self.args['model_type'] in ['bert', 'xlnet'] else None,
                          # XLM does not  use segment_ids
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

        if self.args['output_mode'] == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.args['output_mode'] == "regression":
            preds = np.squeeze(preds)

        print("out_label_ids", out_label_ids)
        result = self.get_eval_report(out_label_ids, preds)
        results.update(result)

        return results

    def load_and_cache_examples(self, task, tokenizer, evaluate):

        self.processor = processors[task]()
        output_mode = self.args['output_mode']

        if not self.features_exists and not self.args['reprocess_input_data']:
            self.logger.info("Loading features from cached file %s")


        else:
            self.features_exists = True
            label_list = self.processor.get_labels()

            if evaluate:
                examples = self.processor.get_dev_examples(self.eval_list_data)
            else:
                examples = self.processor.get_train_examples(self.train_list_data)

            self.features = convert_examples_to_features(examples, label_list, self.args['max_seq_length'], tokenizer,
                                                    output_mode,
                                                    cls_token_at_end=bool(self.args['model_type'] in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if self.args['model_type'] in [
                                                        'xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(self.args['model_type'] in ['roberta']),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(self.args['model_type'] in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if self.args['model_type'] in [
                                                        'xlnet'] else 0)

        all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        all_label_ids = None
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in self.features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in self.features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
