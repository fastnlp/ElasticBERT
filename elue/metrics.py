import warnings

from transformers.file_utils import is_sklearn_available, requires_backends


if is_sklearn_available():
    from sklearn.metrics import f1_score, matthews_corrcoef

    from scipy.stats import pearsonr, spearmanr


def simple_accuracy(preds, labels):
    requires_backends(simple_accuracy, "sklearn")
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    requires_backends(acc_and_f1, "sklearn")
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    requires_backends(pearson_and_spearman, "sklearn")
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def elue_compute_metrics(task_name, preds, labels):
    requires_backends(elue_compute_metrics, "sklearn")
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "snli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "scitail":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "imdb":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)