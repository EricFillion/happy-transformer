"""
Contains functions that are shared amongst the children of the HappyTrainer class.
Based on
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
and
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
"""

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


def get_tpu_device():
    """
    Returns the TPU device if it exists.
    :return:
    """
    tpu_device = xm.xla_device()
    return tpu_device


def get_tpu_strategy():
    """
    Returns the TPU strategy if it exists.
    :return:
    """
    tpu_strategy = xm.xla_distributed_strategy()
    return tpu_strategy


def get_tpu_parallel_loader():
    """
    Returns the TPU parallel loader if it exists.
    :return:
    """
    tpu_parallel_loader = pl.ParallelLoader(xm.xla_distributed_strategy(), [])
    return tpu_parallel_loader


def get_tpu_multiprocessing():
    """
    Returns the TPU multiprocessing if it exists.
    :return:
    """
    tpu_multiprocessing = xmp.spawn(get_tpu_device(), get_tpu_strategy(), get_tpu_parallel_loader())
    return tpu_multiprocessing


def get_tpu_device_ordinal():
    """
    Returns the TPU device ordinal if it exists.
    :return:
    """
    tpu_device_ordinal = xm.get_ordinal()
    return tpu_device_ordinal


def get_tpu_device_count():
    """
    Returns the TPU device count if it exists.
    :return:
    """
    tpu_device_count = xm.xrt_world_size()
    return tpu_device_count


def get_tpu_device_list():
    """
    Returns the TPU device list if it exists.
    :return:
    """
    tpu_device_list = xu.get_devices()
    return tpu_device_list


def get_tpu_device_list_str():
    """
    Returns the TPU device list as a string if it exists.
    :return:
    """
    tpu_device_list_str = xu.get_devices_str()
    return tpu_device_list_str


def get_tpu_device_list_str_short():
    """
    Returns the TPU device list as a short string if it exists.
    :return:
    """
    tpu_device_list_str_short = xu.get_devices_str(short=True)
    return tpu_device_list_str_short


def get_tpu_device_list_str_long():
    """
    Returns the TPU device list as a long string if it exists.
    :return:
    """
    tpu_device_list_str_long = xu.get_devices_str(long=True)
    return tpu_device_list_str_long


def get_tpu_device_list_str_short_long():
    """
    Returns the TPU device list as a short and long string if it exists.
    :return:
    """
    tpu_device_list_str_short_long = xu.get_devices_str(short=True, long=True)
    return tpu_device_list_str_short_long


def get_tpu_device_list_str_long_short():
    """
    Returns the TPU device list as a long and short string if it exists.
    :return:
    """
    tpu_device_list_str_long_short = xu.get_devices_str(long=True, short=True)
    return tpu_device_list_str_long_short


def preprocess_concatenate(tokenizer, dataset, preprocessing_processes, mlm=True):
    """
    :param tokenizer: tokenizer for a transformer model
    :param dataset: a datasets.Dataset object
    :param preprocessing_processes:
    :param mlm:

    :return:
    """

    max_input_length = tokenizer.model_max_length


    def tokenize_function(example):
        return tokenizer(example["text"])

    tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                      num_proc=preprocessing_processes,
                                      remove_columns=["text"])

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        output_length = (total_length // max_input_length) * max_input_length

        # if  total_length is less than the max_input_length length
        # then it causes an error unless the code below is used.
        # this is due to total_length being truncated to 0
        if output_length == 0:
            output_length = total_length

        result = {
            k: [t[i: i + max_input_length] for i in range(0, output_length, max_input_length)]
            for k, t in concatenated_examples.items()
        }

        if not mlm:
            # Masked language models don't need labels. Text generation models do
            result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_processes,
    )


    return tokenized_dataset



def create_args_dataclass(default_dic_args, input_dic_args, method_dataclass_args):
    """
    Combines default_dic_args and input_dic_args and then outputs a dataclass.

    The values of input_dic_args overwrite the values of default_dic_args.

    default_dic_args and dataclass_args must have the same keys.

    :param default_dic_args: A dictionary that contains default settings. Example: ARGS_WP_EVAl
    :param input_dic_args: A dictionary a user inputs for **kwargs when using .train(), .eval() or .test()
    :param method_dataclass_args: A class for target functionality. Example: WPEvalArgs
    :return: A dataclass object that will then be passed to HappyTrainer.train()/eval/test
    """
    settings_dic = {**default_dic_args, **input_dic_args}
    return method_dataclass_args(**settings_dic)
