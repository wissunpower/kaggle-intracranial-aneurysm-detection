
import numpy as np

from sklearn.metrics import roc_auc_score

import torch


class ParticipantVisibleError(Exception):
    pass


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # _, predicted = torch.max(logits, -1)
    predicted = torch.argmax(logits, dim=-1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def calculate_logits_accuracy(preds: np.ndarray, labels: np.ndarray
                              , threshold_middle: float=0.5, threshold_margin = 0.0625
                              , for_mean: bool=False, auc_weights: list[float]|None = None) -> float:
    logit_size = preds.shape[-1]

    # Handle weights
    if auc_weights is None:  # Uniform weights
        weights_array = np.ones(logit_size)
    else:
        weights_array = np.asarray(auc_weights)

    # Check weight dimensions
    if len(weights_array) != logit_size:
        raise ValueError(
            f'Number of weights ({len(weights_array)}) must match '
            f'number of classes ({logit_size})'
        )

    # Check for non-negative weights
    if np.any(weights_array < 0):
        raise ValueError('All class weights must be non-negative')

    # Check that at least one weight is positive
    if np.sum(weights_array) == 0:
        raise ValueError('At least one class weight must be positive')

    # Normalize weights to sum to 1
    weights_array = weights_array / np.sum(weights_array)

    if preds.shape != labels.shape:
        print(f'Not match preds and labels shape, '
              f'logits.shape: {preds.shape}, labels.shape: {labels.shape}')
        raise ValueError()

    preds = preds.reshape(-1, logit_size)
    labels = labels.reshape(-1, logit_size)
    logit_count = preds.shape[0]

    threshold_margin = max(threshold_margin, 0.)
    threshold_upper = min(threshold_middle + threshold_margin, 1)
    threshold_lower = max(threshold_middle - threshold_margin, 0)

    weights_array = np.expand_dims(weights_array, axis=0)
    weights_array = np.repeat(weights_array, logit_count, axis=0)

    true_positive_weights = np.where(preds > threshold_upper, weights_array, 0)
    true_negative_weights = np.where(preds < threshold_lower, weights_array, 0)

    positive_labels_mask = np.where(labels >= 1.0, 1, 0)
    negative_labels_mask = np.where(labels <= 0.0, 1, 0)

    positive_valid_weights = np.sum(true_positive_weights * positive_labels_mask)
    negative_valid_weights = np.sum(true_negative_weights * negative_labels_mask)

    accuracy = positive_valid_weights + negative_valid_weights

    if for_mean:
        accuracy /= logit_count

    return accuracy

def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, average=None) -> float:
    """Return 0.5 when class has only positives or only negatives (avoid exceptions)"""
    try:
        if np.unique(y_true).size < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score, average=average))
    except Exception:
        return 0.5

def compute_final_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Final = 0.5 * ( AUC_AP + mean(13 location AUCs) )"""

    num_class: int = y_true.shape[-1]
    loc_idx = list(range(num_class - 1))
    ap_idx = num_class - 1
    
    auc_loc = [_safe_roc_auc(y_true[:, i], y_prob[:, i]) for i in loc_idx]
    auc_ap = _safe_roc_auc(y_true[:, ap_idx], y_prob[:, ap_idx])
    mean_loc = float(np.mean(auc_loc)) if len(auc_loc) > 0 else 0.5
    final_score = 0.5 * (auc_ap + mean_loc)
    
    return final_score


def weighted_multilabel_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_weights: list[float]|None = None,
) -> float:
    """Compute weighted AUC for multilabel classification.

    Parameters:
    -----------
    y_true : np.ndarray of shape (n_samples, n_classes)
        True binary labels (0 or 1) for each class
    y_scores : np.ndarray of shape (n_samples, n_classes)
        Target scores (probability estimates or decision values)
    class_weights : array-like of shape (n_classes,), optional
        Weights for each class. If None, uniform weights are used.
        Weights will be normalized to sum to 1.

    Returns:
    --------
    weighted_auc : float
        The weighted average AUC

    Raises:
    -------
    ValueError
        If any class does not have both positive and negative samples
    """
    n_classes = y_true.shape[-1]
    y_true = np.asarray(y_true).reshape(-1)
    y_scores = np.asarray(y_scores).reshape(-1)

    # Get AUC for each class
    try:
        individual_aucs = _safe_roc_auc(y_true, y_scores, average=None)
    except ValueError:
        raise ParticipantVisibleError(
            'AUC could not be calculated from given predictions.'
        ) from None

    # Handle weights
    if class_weights is None:  # Uniform weights
        weights_array = np.ones(n_classes)
    else:
        weights_array = np.asarray(class_weights)

    # Check weight dimensions
    if len(weights_array) != n_classes:
        raise ValueError(
            f'Number of weights ({len(weights_array)}) must match '
            f'number of classes ({n_classes})'
        )

    # Check for non-negative weights
    if np.any(weights_array < 0):
        raise ValueError('All class weights must be non-negative')

    # Check that at least one weight is positive
    if np.sum(weights_array) == 0:
        raise ValueError('At least one class weight must be positive')

    # Normalize weights to sum to 1
    weights_array = weights_array / np.sum(weights_array)

    # Compute weighted average
    return np.sum(individual_aucs * weights_array)


def weighted_multilabel_auc_for_multiset(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_weights: list[float]|None = None,
) -> list[float]:
    logit_size = y_scores.shape[-1]

    y_scores = y_scores.reshape(-1, logit_size)
    y_true = y_true.reshape(-1, logit_size)

    return [weighted_multilabel_auc(true, scores, class_weights)
            for true, scores in zip(y_true, y_scores) if 0 < np.sum(true)]

if __name__ == '__main__':
    
    y_true = np.array([0., 1., 0., 1.])
    y_score = np.array([0.1, 0.9, 0.3, 0.7])
    roc_auc = roc_auc_score(y_true, y_score, average=None)

    # y_true = np.array([0., 0., 0., 0., 1.,
    #                     0., 0., 0., 0., 0.,
    #                     0., 0., 0., 1.,])
    y_true = np.array([0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.,
                        0., 0., 0., 0.,])
    # y_pred = np.array([0., 0., 0., 0., 0.,
    #                     0., 1., 0., 0., 0.,
    #                     0., 0., 0., 1.,])
    y_pred = np.array([0., 0.5, 0., 0., 0.4,
                        0., 0.2, 0.1, 0.2, 0.,
                        0., 0.3, 0., 0.7,])
    label_auc_weights = [1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1.,
                         1., 1., 1., 13.,]
    
    y2_true = np.array([[0., 0., 0., 0., 1.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 1.,],
                        [0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0.,]])
    y2_pred = np.array([[0., 0.5, 0., 0., 0.4,
                         0., 0.2, 0.1, 0.2, 0.,
                         0., 0.3, 0., 0.7,],
                        [0., 0.5, 0., 0.8, 0.4,
                         0., 0., 0.1, 0.2, 0.,
                         0., 0.3, 0., 0.4,]])

    accuracy = weighted_multilabel_auc_for_multiset(y2_true, y2_pred, label_auc_weights)
    
    # accuracy = calculate_logits_accuracy(y2_pred, y2_true, auc_weights=label_auc_weights)
    print(f'accuracy: {accuracy}')
