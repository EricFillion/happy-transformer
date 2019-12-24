import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange
from transformers import AdamW, BertForMaskedLM, BertTokenizer
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup


class TextDataset(Dataset):
    """
    Used to turn .txt file into a suitable dataset object
    """

    def __init__(self, tokenizer, file_path, block_size=512):
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer.encode(text, add_special_tokens=True)  # Get ids from text
        self.examples = []
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate examples to a max blocksize
            self.examples.append(tokenized_text[i:i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def set_seed(seed=42):
    """
    Sets seed for all random number generators available.
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        print('Cuda manual seed is not set')


def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    * The standard implementation from Huggingface Transformers library *
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)  # MLM Prob is 0.15 in examples
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(train_dataset, model, tokenizer, batch_size=1, lr=5e-5, adam_epsilon=1e-8, epochs=1):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    t_total = len(train_dataloader) // batch_size  # Total Steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=0, t_total=t_total)

    # ToDo Case for fp16

    # Start of training loop
    model.train()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    train_iterator = trange(int(epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            inputs, labels = mask_tokens(batch, tokenizer)
            inputs = inputs.to(device)  # Don't bother if you don't have a gpu
            labels = labels.to(device)

            outputs = model(inputs, masked_lm_labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss.backward()
            tr_loss += loss.item()

            #             if (step + 1) % 1 == 0: # 1 here is a placeholder for gradient accumulation steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

    model_directory = save_model(model, tokenizer, 'model')
    logger.info("Model saved at %s", model_directory)

    return global_step, tr_loss / global_step


def save_model(model: object, tokenizer: object, output_dir: str):
    """
    Saves the model and the tokenizer to the specified output
    directory.

    :param model:
    :param tokenizer:
    :param output_dir:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    saved_model = model.save_pretrained(output_dir)
    saved_tokenizer = tokenizer.save_pretrained(output_dir)
    return output_dir


def switch_to_new(output_dir):
    """
    Switches to the newly created model/tokenizer in the output
    directory.

    :param output_dir:
    :return:
    """
    model = BertForMaskedLM.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    return model, tokenizer


def load_and_cache_examples(tokenizer, file_path, block_size=512):
    """
    Creates a dataset object from file path.
    :param tokenizer:
    :param file_path:
    :param block_size:
    :return:
    """
    dataset = TextDataset(tokenizer, file_path=file_path, block_size=block_size)
    return dataset


def evaluate(model, tokenizer, eval_dataset, batch_size=1):
    """

    :param model:
    :param tokenizer:
    :param eval_dataset:
    :param batch_size:
    :return:
    """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    # Evaluation loop

    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        inputs, labels = mask_tokens(batch, tokenizer)
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        'perplexity': perplexity
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return result
