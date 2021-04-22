"""
Contains functions that are shared amongst the children of the HappyTrainer class.
"""
def preprocess_lm_data(tokenizer, datasets, args):
    """
    :param tokenizer: tokenizer for a transformer model
    :param datasets: a datasets.Dataset object
    :param args: A dictionary that contains settings
    :return:
    """

    def tokenize_function(example):
        return tokenizer(example["text"],
                         add_special_tokens=True, truncation=True,)

    tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                      num_proc=args["preprocessing_processes"],
                                      remove_columns=["text"])

    return tokenized_datasets
