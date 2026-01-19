import numpy as np


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, positive_class=1):
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
    return tp / (tp + fp + 1e-10)


def recall(y_true, y_pred, positive_class=1):
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
    return tp / (tp + fn + 1e-10)


def f1_score(y_true, y_pred, positive_class=1):
    p = precision(y_true, y_pred, positive_class)
    r = recall(y_true, y_pred, positive_class)
    return 2 * p * r / (p + r + 1e-10)


def confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = np.sum(
                (y_true == true_label) & (y_pred == pred_label)
            )

    return matrix


def classification_report(y_true, y_pred):
    return {
        "Accuracy": accuracy(y_true, y_pred),
        "Precision": precision(y_true, y_pred),
        "Recall": recall(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }
