import os
import sys
import logging
sys.path.append('../')

import torch
from torch.utils.data import TensorDataset

from transformers import glue_convert_examples_to_features
from transformers import glue_output_modes
from transformers import glue_processors
from transformers.trainer_utils import is_main_process

from elue import (
    elue_output_modes,
    elue_processors,
    elue_convert_examples_to_features,
)

logger = logging.getLogger(__name__)


def load_and_cache_examples_glue(args, task, tokenizer, data_type="train"):
    if args.local_rank not in [-1, 0] and data_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = glue_processors[task]()
    output_mode = glue_output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            data_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif data_type == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise NotImplementedError

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not data_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = None
    if data_type != "test":
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        

    return dataset


def load_and_cache_examples_elue(args, task, tokenizer, data_type="train"):
    if args.local_rank not in [-1, 0] and data_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = elue_processors[task]()
    output_mode = elue_output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            data_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif data_type == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise NotImplementedError

        features = elue_convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not data_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = None
    if data_type != "test":
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

    return dataset
