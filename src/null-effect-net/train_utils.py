from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import random
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

import models


def model_forward_pass(model, data, query_node_indices, labels):
    if isinstance(model, models.MLPClassifier):
        q_feats = data['q_feats'][query_node_indices]
        active_feats = data['active_feats'][query_node_indices]
        logits = model(q_feats, active_feats)
    elif isinstance(model, models.GNNClassifier):
        logits = model(data, query_node_indices)
    elif isinstance(model, models.GNNAttentionClassifier):
        logits = model(data, query_node_indices)
    else:
        raise ValueError("Unsupported model type")
    
    # Compute the loss
    loss = model.compute_loss(logits, labels)
    
    return logits, loss

# Training loop for one epoch
def train_one_epoch(model, dataloader, optimizer, device, data=None):
    model.train()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    class_counts = {0: 0, 1: 0}

    for batch_idx, (query_node_indices, active_feats, labels) in enumerate(dataloader):
        query_node_indices = query_node_indices.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        logits, loss = model_forward_pass(model, data, query_node_indices, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Predictions & metrics tracking
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)

        total_loss += loss.item()

        # Track class distribution
        for label in labels.cpu().numpy():
            class_counts[int(label)] += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"[Batch {batch_idx+1}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    metrics['loss'] = avg_loss

    print(f"\n[Training] Loss: {avg_loss:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"Class distribution: 0 -> {class_counts[0]}, 1 -> {class_counts[1]}")

    return metrics


# Evaluation loop (no gradients)
def evaluate(model, data, dataloader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    class_counts = {0: 0, 1: 0}

    with torch.no_grad():
        for query_node_indices, labels in dataloader:
            query_node_indices = query_node_indices.to(device)
            labels = labels.to(device)

            logits, _ = model_forward_pass(model, data, query_node_indices, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

            for label in labels.cpu().numpy():
                class_counts[int(label)] += 1

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))

    print(f"\n[Evaluation] Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"Class distribution: 0 -> {class_counts[0]}, 1 -> {class_counts[1]}")

    return metrics, y_true, y_pred, y_prob


def train_one_epoch_mlp(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    class_counts = {0: 0, 1: 0}

    for batch_idx, (query_feats, labels) in enumerate(dataloader):
        query_feats = query_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(query_feats)
        loss = model.compute_loss(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Predictions & metrics tracking
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)

        total_loss += loss.item()

        # Track class distribution
        for label in labels.cpu().numpy():
            class_counts[int(label)] += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"[Batch {batch_idx+1}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    metrics['loss'] = avg_loss

    print(f"\n[Training] Loss: {avg_loss:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"Class distribution: 0 -> {class_counts[0]}, 1 -> {class_counts[1]}")

    return metrics

# Evaluation loop (no gradients)
def evaluate_mlp(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    class_counts = {0: 0, 1: 0}

    with torch.no_grad():
        for query_feats, labels in dataloader:
            query_feats = query_feats.to(device)
            labels = labels.to(device)

            logits = model(query_feats)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

            for label in labels.cpu().numpy():
                class_counts[int(label)] += 1

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))

    print(f"\n[Evaluation] Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"Class distribution: 0 -> {class_counts[0]}, 1 -> {class_counts[1]}")

    return metrics, y_true, y_pred, y_prob


def train_one_epoch_gnn(model, data, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    class_counts = {0: 0, 1: 0}

    for batch_idx, (query_node_indices, labels, active_node_indices) in enumerate(dataloader):
        query_node_indices = query_node_indices.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(data, query_node_indices, active_node_indices)
        loss = model.compute_loss(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Predictions & metrics tracking
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)

        total_loss += loss.item()

        # Track class distribution
        for label in labels.cpu().numpy():
            class_counts[int(label)] += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"[Batch {batch_idx+1}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    metrics['loss'] = avg_loss

    print(f"\n[Training] Loss: {avg_loss:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"Class distribution: 0 -> {class_counts[0]}, 1 -> {class_counts[1]}")

    return metrics

# Evaluation loop (no gradients)
def evaluate_gnn(model, data, dataloader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    class_counts = {0: 0, 1: 0}

    with torch.no_grad():
        for query_node_indices, labels, active_node_indices in dataloader:
            query_node_indices = query_node_indices.to(device)
            labels = labels.to(device)

            logits = model(data, query_node_indices)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

            for label in labels.cpu().numpy():
                class_counts[int(label)] += 1

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))

    print(f"\n[Evaluation] Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"Class distribution: 0 -> {class_counts[0]}, 1 -> {class_counts[1]}")

    return metrics, y_true, y_pred, y_prob

def evaluate_gnn_with_attention(model, data, dataloader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    attn_weights_all = []
    query_node_all = []
    set_mask_all = []

    with torch.no_grad():
        for query_node_indices, labels, active_node_indices in dataloader:
            query_node_indices = query_node_indices.to(device)
            labels = labels.to(device)

            logits, attn_weights = model(data, query_node_indices, active_node_indices, return_attn=True)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

            attn_weights_all.append(attn_weights.cpu())
            query_node_all.append(query_node_indices.cpu())
            set_mask_all.append(data.set_mask.cpu())

    metrics = compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    print(f"\n[Evaluation] Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")

    return metrics, y_true, y_pred, y_prob, {
        "attn_weights": torch.cat(attn_weights_all),
        "query_nodes": torch.cat(query_node_all),
        "set_masks": torch.cat(set_mask_all),
    }


def compute_metrics(y_true, y_pred, y_prob):

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    return {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

def bootstrap_evaluation(model, dataloader, device, n_bootstrap=1000):
    """Returns confidence intervals."""
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in dataloader:
            q_feats, active_feats, labels = batch
            q_feats, labels = q_feats.to(device), labels.to(device)
            active_feats = [af.to(device) for af in active_feats]

            logits = model(q_feats, active_feats)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Bootstrap AUC scores
    bootstrapped_aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        try:
            auc = roc_auc_score(y_true[idx], y_prob[idx])
        except:
            auc = 0.5
        bootstrapped_aucs.append(auc)

    # Compute confidence intervals
    ci_lower = np.percentile(bootstrapped_aucs, 2.5)
    ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    return np.mean(bootstrapped_aucs), (ci_lower, ci_upper)


def save_model(model, dir, experiment=None):
    model_type = model.__class__.__name__
    now = datetime.datetime.now()
    date_time_string = now.strftime("%m_%d-%H_%M")
    if experiment:
        os.makedirs(os.path.join(dir, experiment), exist_ok=True)
        new_model_dir = model_type
        new_model_dir_fullpath = os.path.join(dir, experiment, new_model_dir)
        os.makedirs(new_model_dir_fullpath, exist_ok=True)
    else:
        new_model_dir = model_type
        new_model_dir_fullpath = os.path.join(dir, new_model_dir)
        os.makedirs(new_model_dir_fullpath, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(new_model_dir_fullpath, f'checkpoint_{date_time_string}.pt'))

def load_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def plot_precision_recall(y_true, y_prob, title='Precision-Recall Curve'):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_prediction_histograms(y_true, y_prob, title='Prediction Confidence by Class'):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    plt.figure(figsize=(7, 5))
    
    # Plot histograms for true class 0 and true class 1 separately
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label='True Class 0', color='blue')
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label='True Class 1', color='orange')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()