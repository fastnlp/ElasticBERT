import logging
import os
import csv
import sys
sys.path.append('../')

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import glue_compute_metrics

from elue import elue_compute_metrics

from load_data import (
    load_and_cache_examples_glue,
    load_and_cache_examples_elue,
)

logger = logging.getLogger(__name__)


def evaluate_glue_patience(args, model, tokenizer, prefix="", patience=0):
    model.elasticbert.set_regression_threshold(args.regression_threshold)
    model.elasticbert.set_patience(patience)
    model.elasticbert.reset_stats()

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    results_all = []
    exit_layer = []
    for i in range(args.num_hidden_layers):
        results_all.append({})
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples_glue(args, eval_task, tokenizer, data_type='dev')

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
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        preds_all = []
        for i in range(args.num_hidden_layers):
            preds_all.append(None)
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if out_label_ids is None:
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            if patience == 0:
                for i, pred in enumerate(preds_all):
                    if pred is None:
                        preds_all[i] = logits[i].detach().cpu().numpy()
                    else:
                        preds_all[i] = np.append(pred, logits[i].detach().cpu().numpy(), axis=0)
            else:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            if patience == 0:
                for i, pred in enumerate(preds_all):
                    preds_all[i] = np.argmax(pred, axis=1)
            else:
                preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            if patience == 0:
                for i, pred in enumerate(preds_all):
                    preds_all[i] = np.squeeze(pred)
            else:
                preds = np.squeeze(preds)

        if patience == 0:
            for i, pred in enumerate(preds_all):
                result = glue_compute_metrics(eval_task, pred, out_label_ids)
                results_all[i].update(result)

        else:
            result = glue_compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                print("  %s = %s" % (key, str(result[key])))

            exiting_layer_every_ins = model.elasticbert.exiting_layer_every_ins
            exit_layer.append(exiting_layer_every_ins)
            
            if args.task_name == "mnli":
                model.elasticbert.exiting_layer_every_ins = []

    if patience != 0:
        speed_up = model.elasticbert.log_stats()
        return results, speed_up, exit_layer

    return results_all



def evaluate_elue_patience(args, model, tokenizer, prefix="", patience=0):
    model.elasticbert.set_regression_threshold(args.regression_threshold)
    model.elasticbert.set_patience(patience)
    model.elasticbert.reset_stats()

    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    results_all = []
    exit_layer = []
    for i in range(args.num_hidden_layers):
        results_all.append({})

    eval_dataset = load_and_cache_examples_elue(args, eval_task, tokenizer, data_type='dev')

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
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    preds_all = []
    for i in range(args.num_hidden_layers):
        preds_all.append(None)
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if out_label_ids is None:
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        if patience == 0:
            for i, pred in enumerate(preds_all):
                if pred is None:
                    preds_all[i] = logits[i].detach().cpu().numpy()
                else:
                    preds_all[i] = np.append(pred, logits[i].detach().cpu().numpy(), axis=0)
        else:
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)


    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        if patience == 0:
            for i, pred in enumerate(preds_all):
                preds_all[i] = np.argmax(pred, axis=1)
        else:
            preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        if patience == 0:
            for i, pred in enumerate(preds_all):
                preds_all[i] = np.squeeze(pred)
        else:
            preds = np.squeeze(preds)

    if patience == 0:
        for i, pred in enumerate(preds_all):
            result = elue_compute_metrics(eval_task, pred, out_label_ids)
            results_all[i].update(result)

    else:
        result = elue_compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %s" % (key, str(result[key])))

        exiting_layer_every_ins = model.elasticbert.exiting_layer_every_ins
        exit_layer.append(exiting_layer_every_ins)

    if patience != 0:
        speed_up = model.elasticbert.log_stats()
        return results, speed_up, exit_layer

    return results_all


def evaluate_glue_entropy(args, model, tokenizer, prefix="", eval_highway=False, entropy=0.):
    model.elasticbert.set_early_exit_entropy(entropy)
    model.elasticbert.set_eval_state(eval_highway)
    model.elasticbert.reset_stats()
        
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    results_all = []
    exit_layer = []
    for i in range(args.num_hidden_layers):
        results_all.append({})
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples_glue(args, eval_task, tokenizer, data_type='dev')

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
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        preds_all = []
        for i in range(args.num_hidden_layers):
            preds_all.append(None)
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[-1],
                }
                inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if out_label_ids is None:
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            if not eval_highway:
                for i, pred in enumerate(preds_all):
                    if pred is None:
                        preds_all[i] = logits[i].detach().cpu().numpy()
                    else:
                        preds_all[i] = np.append(pred, logits[i].detach().cpu().numpy(), axis=0)
            else:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            if not eval_highway:
                for i, pred in enumerate(preds_all):
                    preds_all[i] = np.argmax(pred, axis=1)
            else:
                preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            if not eval_highway:
                for i, pred in enumerate(preds_all):
                    preds_all[i] = np.squeeze(pred)
            else:
                preds = np.squeeze(preds)

        if not eval_highway:
            for i, pred in enumerate(preds_all):
                result = glue_compute_metrics(eval_task, pred, out_label_ids)
                results_all[i].update(result)

        else:
            result = glue_compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                print("  %s = %s" % (key, str(result[key])))

            exiting_layer_every_ins = model.elasticbert.exiting_layer_every_ins
            exit_layer.append(exiting_layer_every_ins)

            if args.task_name == "mnli":
                model.elasticbert.exiting_layer_every_ins = []

    if eval_highway:
        speed_up = model.elasticbert.log_stats()
        return results, speed_up, exit_layer

    return results_all


def evaluate_elue_entropy(args, model, tokenizer, prefix="", eval_highway=False, entropy=0.):
    model.elasticbert.set_early_exit_entropy(entropy)
    model.elasticbert.set_eval_state(eval_highway)
    model.elasticbert.reset_stats()     

    eval_task = args.task_name
    eval_output_dir = args.output_dir  

    results = {}
    results_all = []
    exit_layer = []
    for i in range(args.num_hidden_layers):
        results_all.append({})

    eval_dataset = load_and_cache_examples_elue(args, eval_task, tokenizer, data_type='dev')

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
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    preds_all = []
    for i in range(args.num_hidden_layers):
        preds_all.append(None)
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[-1],
            }
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if out_label_ids is None:
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        if not eval_highway:
            for i, pred in enumerate(preds_all):
                if pred is None:
                    preds_all[i] = logits[i].detach().cpu().numpy()
                else:
                    preds_all[i] = np.append(pred, logits[i].detach().cpu().numpy(), axis=0)
        else:
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        if not eval_highway:
            for i, pred in enumerate(preds_all):
                preds_all[i] = np.argmax(pred, axis=1)
        else:
            preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        if not eval_highway:
            for i, pred in enumerate(preds_all):
                preds_all[i] = np.squeeze(pred)
        else:
            preds = np.squeeze(preds)

    if not eval_highway:
        for i, pred in enumerate(preds_all):
            result = elue_compute_metrics(eval_task, pred, out_label_ids)
            results_all[i].update(result)

    else:
        result = elue_compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %s" % (key, str(result[key])))

        exiting_layer_every_ins = model.elasticbert.exiting_layer_every_ins
        exit_layer.append(exiting_layer_every_ins)

    if eval_highway:
        speed_up = model.elasticbert.log_stats()
        return results, speed_up, exit_layer

    return results_all