import os
import csv
import sys
import logging

sys.path.append('../')

import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers import glue_processors

from elue import elue_compute_metrics, elue_processors

from load_data import (
    load_and_cache_examples_glue,
    load_and_cache_examples_elue,
)

logger = logging.getLogger(__name__)


def inference_glue(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = glue_processors[eval_task]()
        label_list = processor.get_labels()
        eval_dataset = load_and_cache_examples_glue(args, eval_task, tokenizer, data_type="test")

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running Inference  *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        preds = None

        for batch in tqdm(eval_dataloader, desc="Infering"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                logits = outputs[1]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        output_infer_file = os.path.join(eval_output_dir, "{}.tsv".format(eval_task))
        with open(output_infer_file, "w", encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t', quotechar=None)
            writer.writerow(["index", "prediction"])
            for i, pred in enumerate(preds):
                if args.output_mode == "classification":
                    prediction = label_list[pred]
                elif args.output_mode == "regression":
                    prediction = str(pred)
                writer.writerow([i, prediction]) 


def inference_elue(args, model, tokenizer):

    eval_task = args.task_name
    eval_output_dir = args.output_dir

    processor = elue_processors[eval_task]()
    label_list = processor.get_labels()
    eval_dataset = load_and_cache_examples_elue(args, eval_task, tokenizer, data_type="test")

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Inference  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    preds = None

    for batch in tqdm(eval_dataloader, desc="Infering"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[1]

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    output_infer_file = os.path.join(eval_output_dir, "{}.tsv".format(eval_task))
    with open(output_infer_file, "w", encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter='\t', quotechar=None)
        writer.writerow(["index", "prediction"])
        for i, pred in enumerate(preds):
            if args.output_mode == "classification":
                prediction = label_list[pred]
            elif args.output_mode == "regression":
                prediction = str(pred)
            writer.writerow([i, prediction])
