from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(
    labels,
    preds,
    probs
):

    metrics = {

        "precision": precision_score(
            labels,
            preds
        ),

        "recall": recall_score(
            labels,
            preds
        ),

        "f1": f1_score(
            labels,
            preds
        ),

        "roc_auc": roc_auc_score(
            labels,
            probs
        ),

        "pr_auc": average_precision_score(
            labels,
            probs
        )
    }

    return metrics
