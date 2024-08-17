import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

# Define classes
classes = ["dx4e", "dx6i", "MTx", "Nineeg", "Parrot", "q205", "S500", "tello", "WiFi", "wltoys"]
#classes = ["drone", "wifi"]

# Specify the directories where the history and confusion matrix files are stored
history_dir = r'C:\Users\lolaf\Desktop\Thesis_code\Results\Conv1D\History'  # Update this path if necessary
confusion_matrix_dir = r'C:\Users\lolaf\Desktop\Thesis_code\Results\Conv1D\Confusion'  # Update this path if necessary
pr_curve_data_dir = r'C:\Users\lolaf\Desktop\Thesis_code\Results\Conv1D\Graphs'  # Update this path if necessary

# List all history files, confusion matrix files, and precision-recall curve data files
history_files = [f for f in os.listdir(history_dir) if f.startswith('history') and f.endswith('.pkl')]
confusion_matrix_files = [f for f in os.listdir(confusion_matrix_dir) if f.startswith('confusion_matrix') and f.endswith('.pkl')]
pr_curve_data_files = [f for f in os.listdir(pr_curve_data_dir) if f.startswith('pr_curve_data') and f.endswith('.pkl')]

# Load histories, confusion matrices, and precision-recall curve data
histories = {}
confusion_matrices = {}
pr_curve_data = {}

for file in history_files:
    with open(os.path.join(history_dir, file), 'rb') as f:
        key = file.replace('history_', '').replace('.pkl', '')
        histories[key] = pickle.load(f)

for file in confusion_matrix_files:
    with open(os.path.join(confusion_matrix_dir, file), 'rb') as f:
        key = file.replace('confusion_matrix_', '').replace('.pkl', '')
        confusion_matrices[key] = pickle.load(f)

for file in pr_curve_data_files:
    with open(os.path.join(pr_curve_data_dir, file), 'rb') as f:
        key = file.replace('pr_curve_data_', '').replace('.pkl', '')
        pr_curve_data[key] = pickle.load(f)

def extract_hyperparams_from_key(key):
    # Extract the hyperparameters from the key assuming format 'lr{lr}_bs{batch_size}_pca{pca_components}'
    if key.startswith('lr') and '_bs' in key and '_pca' in key:
        parts = key.split('_')
        lr = parts[0].replace('lr', '')
        batch_size = parts[1].replace('bs', '')
        pca_components = parts[2].replace('pca', '')
        return lr, batch_size, pca_components
    else:
        print(f"Unexpected key format: {key}")
        return None, None, None

# Plot training and validation accuracy/loss
def plot_accuracy_loss(histories):
    for key, history in histories.items():
        lr, batch_size, pca_components = extract_hyperparams_from_key(key)
        if lr is None:
            continue

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Model Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')

        plt.suptitle(f'LR: {lr}, Batch Size: {batch_size}, PCA Components: {pca_components}')
        plt.savefig(f'plot_lr{lr}_bs{batch_size}_pca{pca_components}.png')
        plt.close()

# Plot F1 score, precision, and recall
def plot_metrics(histories):
    for key, history in histories.items():
        lr, batch_size, pca_components = extract_hyperparams_from_key(key)
        if lr is None:
            continue

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(history['f1_score'], label='F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('F1 Score')

        plt.subplot(1, 3, 2)
        plt.plot(history['precision'], label='Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
        plt.title('Precision')

        plt.subplot(1, 3, 3)
        plt.plot(history['recall'], label='Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
        plt.title('Recall')

        plt.suptitle(f'LR: {lr}, Batch Size: {batch_size}, PCA Components: {pca_components}')
        plt.savefig(f'metrics_lr{lr}_bs{batch_size}_pca{pca_components}.png')
        plt.close()

# Plot confusion matrix
def plot_confusion_matrices(confusion_matrices):
    for key, cm in confusion_matrices.items():
        lr, batch_size, pca_components = extract_hyperparams_from_key(key)
        if lr is None:
            continue

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix\nLR: {lr}, Batch Size: {batch_size}, PCA Components: {pca_components}')
        plt.savefig(f'confusion_matrix_lr{lr}_bs{batch_size}_pca{pca_components}.png')
        plt.close()

# Plot Precision-Recall curves
def plot_precision_recall_curves(pr_curve_data):
    for key, data in pr_curve_data.items():
        lr, batch_size, pca_components = extract_hyperparams_from_key(key)
        if lr is None:
            continue

        plt.figure(figsize=(8, 6))
        for i in range(len(classes)):
            plt.step(data[i]['recall'], data[i]['precision'], where='post', label=f'Class {classes[i]} (AP = {data[i]["average_precision"]:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve\nLR: {lr}, Batch Size: {batch_size}, PCA Components: {pca_components}')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(f'pr_curve_lr{lr}_bs{batch_size}_pca{pca_components}.png')
        plt.close()

# Compare different learning rates, batch sizes, and PCA components
def plot_comparison(histories, metric_name):
    plt.figure(figsize=(15, 10))
    for key, history in histories.items():
        lr, batch_size, pca_components = extract_hyperparams_from_key(key)
        if lr is None:
            continue

        plt.plot(history[metric_name], label=f'LR: {lr}, BS: {batch_size}, PCA: {pca_components}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()
    plt.title(f'Comparison of {metric_name.replace("_", " ").title()}')
    plt.savefig(f'comparison_{metric_name}.png')
    plt.close()

# Generate all plots
plot_accuracy_loss(histories)
plot_metrics(histories)
plot_confusion_matrices(confusion_matrices)
plot_precision_recall_curves(pr_curve_data)

metrics = ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss', 'f1_score', 'precision', 'recall']
for metric in metrics:
    plot_comparison(histories, metric)