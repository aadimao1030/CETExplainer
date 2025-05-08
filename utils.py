import tensorflow as tf
import numpy as np
import os
import scipy.sparse as sp
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_curve, auc, average_precision_score)
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import random

def check_prediction_stats(pred: tf.Tensor, name: str = "rel_pred"):
    vals = pred.numpy()
    print(f"{name} stats | min: {vals.min():.4f} | max: {vals.max():.4f} |"
          f" mean: {vals.mean():.4f} | std: {vals.std():.4f}")

def check_label_stats(y: np.ndarray):
    total = len(y)
    ones = y.sum()
    zeros = total - ones
    print(f"Label stats | total: {total} | #1: {ones} | #0: {zeros}")


def setup_environment(config):
    os.environ['PYTHONHASHSEED'] = str(config.SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)


def train_step(model, inputs, labels, optimizer, contrastive_module, drug_features, cell_features, main_loss_weight=1.0, contrast_loss_weight=0.1):
    with tf.GradientTape() as tape:
        preds = model(inputs, training=True)
        main_loss = compute_loss(labels, preds)
        contrast_loss = compute_contrastive_loss(drug_features, cell_features, contrastive_module)
        total_loss = main_loss_weight * main_loss + contrast_loss_weight * contrast_loss

    grads = tape.gradient(total_loss, model.trainable_variables + contrastive_module.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables + contrastive_module.trainable_variables))

    return total_loss, main_loss, contrast_loss

def generate_negative_samples(positive_samples, num_nodes, num_relations, full_adj_inputs):
    # full_adj_inputs is the complete adjacency matrix containing all relations
    existing_edges = set(map(tuple, positive_samples))  # Get all positive samples as a set

    negative_samples = []
    
    while len(negative_samples) < len(positive_samples):
        head = np.random.randint(0, num_nodes)
        tail = np.random.randint(0, num_nodes)
        rel = np.random.randint(0, num_relations)  # Randomly select a relation

        if (head, rel, tail) not in existing_edges:  # Negative samples must not exist in positive samples
            negative_samples.append((head, rel, tail))
    
    return negative_samples

def build_adj_from_edges(edges, num_nodes, num_rel):
    """
    edges: numpy array of shape [m,3], elements may be float, should be converted to int.
    """
    # Create empty sparse matrices for each relation type
    adjs = [sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            for _ in range(num_rel)]
    # Iterate through each edge and force conversion to int
    for e in edges:
        h, r, t = int(e[0]), int(e[1]), int(e[2])
        adjs[r][h, t] = 1

    return [tf.sparse.from_dense(a.toarray()) for a in adjs]

def sparse_tensor_to_triples(sp_tensor, rel_id):
    """
    Convert a (N, N) SparseTensor and its corresponding relation id (int),
    into a numpy array of shape [num_edges, 3]: [head, rel_id, tail]
    """
    # sp_tensor.indices is an [num_edges, 2] int64 Tensor: each row [i, j]
    idx = sp_tensor.indices.numpy()
    num_edges = idx.shape[0]
    # Create a constant vector of [num_edges] filled with rel_id
    rels = np.full((num_edges, 1), rel_id, dtype=np.int32)
    # heads/tails
    heads = idx[:, 0:1].astype(np.int32)
    tails = idx[:, 1:2].astype(np.int32)
    return np.concatenate([heads, rels, tails], axis=1)  # Shape [num_edges,3]


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray):
    """
    Given true labels and predicted scores, calculate:
      - The optimal threshold corresponding to the maximum F1 score
      - AUC-ROC, Average Precision
      - Accuracy, Precision, Recall, F1
      - Confusion Matrix
    Returns: dict(metrics)
    """
    # 1) PR curve, compute precision, recall for each threshold
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
    # precision_recall_curve will return an extra zero-threshold, so we skip the last one
    pr_thresh = pr_thresh[:-1]  # Remove the extra threshold at the end
    prec = prec[:-1]
    rec = rec[:-1]

    # 2) Compute F1 scores for all thresholds
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)

    # 3) Find the index of the maximum F1 score and its corresponding threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)  # Maximize Youden's J
    best_th = thresholds[best_idx]
    print(f"Optimal threshold by Youden: {best_th:.4f}")

    # 4) Use this threshold to perform binary classification
    y_pred = (y_score >= best_th).astype(int)

    # 5) Calculate other metrics
    metrics = {
        'optimal_threshold': best_th,
        'roc_auc': roc_auc_score(y_true, y_score),
        'aupr': average_precision_score(y_true, y_score),
        'accuracy': np.mean(y_pred == y_true),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),  # Directly compute F1 score
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics


def plot_roc_and_aupr(y_true, y_scores, fold):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # AUPR Curve
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(12, 6))

    # Plot ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold} - ROC Curve')
    plt.legend(loc='lower right')

    # Plot AUPR curve
    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='gray', linestyle='--')
    plt.plot(np.linspace(0, 1, 100), np.interp(np.linspace(0, 1, 100), fpr, tpr), color='blue', label=f'AUPR = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Fold {fold} - AUPR Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()


def compute_contrastive_loss(drug_features, cell_features, contrastive_module):
    drug_sim_sparse, cell_sim_sparse = contrastive_module([drug_features, cell_features], training=True)
    contrast_loss = tf.add_n(contrastive_module.losses)
    return contrast_loss

def compute_loss(y_true, y_pred):
    loss_fn = BinaryCrossentropy(from_logits=False)
    return loss_fn(y_true, y_pred)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot the loss and accuracy curves."""
    plt.figure(figsize=(12, 6))

    # Plot loss curve
    for fold in range(1, len(train_losses) + 1):
        plt.subplot(1, 2, 1)
        plt.plot(train_losses[fold - 1], label=f"Fold {fold} Train Loss")
        plt.plot(val_losses[fold - 1], label=f"Fold {fold} Val Loss", linestyle='--')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy curve
    for fold in range(1, len(train_accuracies) + 1):
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies[fold - 1], label=f"Fold {fold} Train Accuracy")
        plt.plot(val_accuracies[fold - 1], label=f"Fold {fold} Val Accuracy", linestyle='--')
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
