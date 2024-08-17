import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

# Define classes
#classes = ["drone", "wifi"]
classes = ["dx4e", "dx6i", "MTx", "Nineeg", "Parrot", "q205", "S500", "tello", "WiFi", "wltoys"]

# Get the current directory of the running script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the filenames for the history and confusion matrix
history_filename = 'history.pkl'
confusion_matrix_filename = 'confusion_matrix.pkl'
y_true_filename = 'y_true.pkl'
y_scores_filename = 'y_scores.pkl'

# Construct the full file paths
history_file_path = os.path.join(current_dir, history_filename)
confusion_matrix_file_path = os.path.join(current_dir, confusion_matrix_filename)
y_true_file_path = os.path.join(current_dir, y_true_filename)
y_scores_file_path = os.path.join(current_dir, y_scores_filename)

# Load history, confusion matrix, true labels, and scores
with open(history_file_path, 'rb') as f:
    history = pickle.load(f)

with open(confusion_matrix_file_path, 'rb') as f:
    confusion_matrix = pickle.load(f)

with open(y_true_file_path, 'rb') as f:
    y_true = pickle.load(f)

with open(y_scores_file_path, 'rb') as f:
    y_scores = pickle.load(f)

# Convert y_true and y_scores to numpy arrays if they are lists
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Print keys to debug
print("History keys:", history.keys())
print("Confusion matrix shape:", confusion_matrix.shape)

# Plot training and validation accuracy/loss
def plot_accuracy_loss(history):
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

    plt.suptitle('Model Training and Validation Metrics')
    plot_filename = 'plot_accuracy_loss.png'
    plt.savefig(os.path.join(current_dir, plot_filename))
    print(f'Saved plot: {plot_filename}')
    plt.close()

#Plot F1 score, precision, and recall
def plot_metrics(history):
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

    plt.suptitle('Model Metrics')
    plot_filename = 'plot_metrics.png'
    plt.savefig(os.path.join(current_dir, plot_filename))
    print(f'Saved plot: {plot_filename}')
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plot_filename = 'confusion_matrix.png'
    plt.savefig(os.path.join(current_dir, plot_filename))
    print(f'Saved plot: {plot_filename}')
    plt.close()

# Plot precision-recall curve
def plot_precision_recall_curve(y_true, y_scores, classes):
    # Binarize the output labels for multi-class precision-recall curve
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))

    # Check dimensions and ensure y_scores are 2D
    if y_scores.ndim == 1:
        raise ValueError("y_scores must be a 2D array for multi-class classification.")

    if y_scores.shape[1] != len(classes):
        raise ValueError("y_scores should have the same number of columns as there are classes.")

    # Compute the precision-recall curve and average precision for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        average_precision = average_precision_score(y_true_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {classes[i]} (AP = {average_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plot_filename = 'precision_recall_curve.png'
    plt.savefig(plot_filename)
    print(f'Saved plot: {plot_filename}')
    plt.show()

# Plot accuracy and loss
plot_accuracy_loss(history)

# Plot metrics
plot_metrics(history)

# Plot confusion matrix
plot_confusion_matrix(confusion_matrix)

# Plot precision-recall curve
plot_precision_recall_curve(y_true, y_scores, classes)
