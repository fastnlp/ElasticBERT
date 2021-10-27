""" elue processors and helpers """

import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union

from datasets import ClassLabel, load_dataset, load_metric
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset

from transformers.file_utils import is_tf_available
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers.data.processors import (
    DataProcessor,
    InputExample,
    InputFeatures,
)


if is_tf_available():
    import tensorflow as tf


logger = logging.get_logger(__name__)

class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


def elue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: ELUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
        task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
        ``InputFeatures`` which can be fed to the model.

    """
    return _elue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


def _elue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if task is not None:
        processor = elue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info(f"Using label list {label_list} for task {task}")
        if output_mode is None:
            output_mode = elue_output_modes[task]
            logger.info(f"Using output mode {output_mode} for task {task}")

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info(f"guid: {example.guid}")
        logger.info(f"features: {features[i]}")

    return features


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (ELUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.tsv')}")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (ELUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ImdbProcessor(DataProcessor):
    """Processor for the SST-2 data set (ELUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (ELUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_a_idx = 1 if set_type == "test" else 0
        text_b_idx = 2 if set_type == "test" else 1
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i - 1}"
            text_a = line[text_a_idx]
            text_b = line[text_b_idx]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 


class SnliProcessor(DataProcessor):
    """Processor for the QNLI data set (ELUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_a_idx = 1 if set_type == "test" else 0
        text_b_idx = 2 if set_type == "test" else 1
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i-1}"
            text_a = line[text_a_idx]
            text_b = line[text_b_idx]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SciTailProcessor(DataProcessor):
    """Processor for the QNLI data set (ELUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_a_idx = 1 if set_type == "test" else 0
        text_b_idx = 2 if set_type == "test" else 1
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i-1}"
            text_a = line[text_a_idx]
            text_b = line[text_b_idx]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


elue_tasks_num_labels = {
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "snli": 3,
    "imdb": 2,
    "scitail": 2,
}

elue_processors = {
    "snli": SnliProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "imdb": ImdbProcessor,
    "scitail": SciTailProcessor,
}

elue_output_modes = {
    "snli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "imdb": "classification",
    "scitail": "classification",
}

