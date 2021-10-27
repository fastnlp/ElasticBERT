import os
import csv
import sys
import logging

sys.path.append('../')

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers import glue_compute_metrics
from transformers import glue_processors

from elue import elue_compute_metrics, elue_processors

from load_data import (
    load_and_cache_examples_glue,
    load_and_cache_examples_elue,
)

logger = logging.getLogger(__name__) 


def inference_elue_patience(args, model, tokenizer, prefix="", patience=0):
    model.elasticbert.set_regression_threshold(args.regression_threshold)
    model.elasticbert.set_patience(patience)
    model.elasticbert.reset_stats()

    eval_task = args.task_name
    eval_output_dir = args.output_dir

    processor = elue_processors[eval_task]()
    label_list = processor.get_labels()
    eval_dataset = load_and_cache_examples_elue(args, eval_task, tokenizer, data_type='test')

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
    logger.info("***** Running inference {} *****".format(prefix))
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
            logits = outputs[0]

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)


    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    output_infer_file = os.path.join(eval_output_dir, "{}_{}.tsv".format(eval_task, str(patience)))
    with open(output_infer_file, "w", encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter='\t', quotechar=None)
        writer.writerow(["index", "prediction"])
        for i, pred in enumerate(preds):
            if args.output_mode == "classification":
                prediction = label_list[pred]
            elif args.output_mode == "regression":
                prediction = str(pred)
            writer.writerow([i, prediction])

    exiting_layer_every_ins = model.elasticbert.exiting_layer_every_ins
    output_infer_file = os.path.join(eval_output_dir, "{}_{}_{}_{}.tsv".format(eval_task, 'infer', 'exit_layer', str(patience)))
    with open(output_infer_file, "w", encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter='\t', quotechar=None)
        writer.writerow(["index", "exit_layer"])
        for i, exit_layer in enumerate(exiting_layer_every_ins):
            writer.writerow([i, exit_layer])

    speed_up = model.elasticbert.log_stats()
    return speed_up


def inference_elue_entropy(args, model, tokenizer, prefix="", eval_highway=False, entropy=0.):
    model.elasticbert.set_early_exit_entropy(entropy)
    model.elasticbert.set_eval_state(eval_highway)
    model.elasticbert.reset_stats()  

    eval_task = args.task_name
    eval_output_dir = args.output_dir  

    eval_dataset = load_and_cache_examples_elue(args, eval_task, tokenizer, data_type='test')

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    processor = elue_processors[eval_task]()
    label_list = processor.get_labels()

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running inference {} *****".format(prefix))
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
            logits = outputs[0]


        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)


    if args.output_mode == "classification":

        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":

        preds = np.squeeze(preds)

    output_infer_file = os.path.join(eval_output_dir, "{}_{}.tsv".format(eval_task, str(entropy)))
    with open(output_infer_file, "w", encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter='\t', quotechar=None)
        writer.writerow(["index", "prediction"])
        for i, pred in enumerate(preds):
            if args.output_mode == "classification":
                prediction = label_list[pred]
            elif args.output_mode == "regression":
                prediction = str(pred)
            writer.writerow([i, prediction])

    exiting_layer_every_ins = model.elasticbert.exiting_layer_every_ins
    output_infer_file = os.path.join(eval_output_dir, "{}_{}_{}_{}.tsv".format(eval_task, 'infer', 'exit_layer', str(entropy)))
    with open(output_infer_file, "w", encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter='\t', quotechar=None)
        writer.writerow(["index", "exit_layer"])
        for i, exit_layer in enumerate(exiting_layer_every_ins):
            writer.writerow([i, exit_layer])

    speed_up = model.elasticbert.log_stats()
    return speed_up


def inference_glue_patience(args, model, tokenizer, prefix="", patience=0):
    model.elasticbert.set_regression_threshold(args.regression_threshold)
    model.elasticbert.set_patience(patience)
    model.elasticbert.reset_stats()

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples_glue(args, eval_task, tokenizer, data_type='test')

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        processor = glue_processors[eval_task]()
        label_list = processor.get_labels()

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running inference {} *****".format(prefix))
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
                logits = outputs[0]


            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)


        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":

            preds = np.squeeze(preds)

        output_infer_file = os.path.join(eval_output_dir, "{}_{}.tsv".format(eval_task, str(patience)))
        with open(output_infer_file, "w", encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t', quotechar=None)
            writer.writerow(["index", "prediction"])
            for i, pred in enumerate(preds):
                if args.output_mode == "classification":
                    prediction = label_list[pred]
                elif args.output_mode == "regression":
                    prediction = str(pred)
                writer.writerow([i, prediction])

        exiting_layer_every_ins = model.elasticbert.exiting_layer_every_ins
        output_infer_file = os.path.join(eval_output_dir, "{}_{}_{}_{}.tsv".format(eval_task, 'infer', 'exit_layer', str(patience)))
        with open(output_infer_file, "w", encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t', quotechar=None)
            writer.writerow(["index", "exit_layer"])
            for i, exit_layer in enumerate(exiting_layer_every_ins):
                writer.writerow([i, exit_layer])
        
        if args.task_name == 'mnli':
            model.elasticbert.exiting_layer_every_ins = []

    speed_up = model.elasticbert.log_stats()
    return speed_up 


def inference_glue_entropy(args, model, tokenizer, prefix="", eval_highway=False, entropy=0.):
    model.elasticbert.set_early_exit_entropy(entropy)
    model.elasticbert.set_eval_state(eval_highway)
    model.elasticbert.reset_stats()  
         
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples_glue(args, eval_task, tokenizer, data_type='test')

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        processor = glue_processors[eval_task]()
        label_list = processor.get_labels()

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running inference {} *****".format(prefix))
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
                logits = outputs[0]


            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)


        if args.output_mode == "classification":

            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":

            preds = np.squeeze(preds)

        output_infer_file = os.path.join(eval_output_dir, "{}_{}.tsv".format(eval_task, str(entropy)))
        with open(output_infer_file, "w", encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t', quotechar=None)
            writer.writerow(["index", "prediction"])
            for i, pred in enumerate(preds):
                if args.output_mode == "classification":
                    prediction = label_list[pred]
                elif args.output_mode == "regression":
                    prediction = str(pred)
                writer.writerow([i, prediction])

        exiting_layer_every_ins = model.elasticbert.exiting_layer_every_ins
        output_infer_file = os.path.join(eval_output_dir, "{}_{}_{}_{}.tsv".format(eval_task, 'infer', 'exit_layer', str(entropy)))
        with open(output_infer_file, "w", encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t', quotechar=None)
            writer.writerow(["index", "exit_layer"])
            for i, exit_layer in enumerate(exiting_layer_every_ins):
                writer.writerow([i, exit_layer])
        
        if args.task_name == 'mnli':
            model.elasticbert.exiting_layer_every_ins = []     

    speed_up = model.elasticbert.log_stats()
    return speed_up 
