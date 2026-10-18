import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Conv1D_LSTM import build_model  # Change the models here for running them
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import pickle
from torchsummary import summary
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure PyTorch uses the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for testing")

# Define classes
classes = ["dx4e", "dx6i", "MTx", "Nineeg", "Parrot", "q205", "S500", "tello", "WiFi", "wltoys"]  # Multi-class classification
num_classes = len(classes)

# Load test data
filename1 = r'C:\Users\lolaf\Desktop\Thesis_code\Data\Resized_128.h5'
filename2 = r'C:\Users\lolaf\Desktop\Thesis_code\Data\GenDataRSNT_10_snrss2_NW.h5'
filename3 = r'C:\Users\lolaf\Desktop\Thesis_code\Data\binary_test_train_values.h5'

h5f = h5py.File(filename1, 'r')
h5f1 = h5py.File(filename2, 'r')
h5f2 = h5py.File(filename3, 'r')

X_test = np.array(h5f['X_test'])
Y_test = np.array(h5f['y_test'])

sig_orig_test = np.array(h5f1['sig_origidx_F_test'])
test_idx = np.array(h5f1['test_idx']).reshape(-1, 1)

h5f.close()
h5f1.close()
h5f2.close()

# Preprocess the data
scaler_sig_orig = StandardScaler()
sig_orig_test = scaler_sig_orig.fit_transform(sig_orig_test)

scaler_idx = StandardScaler()
test_idx = scaler_idx.fit_transform(test_idx)

# Reshape and apply PCA
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
pca_components = 70
pca = PCA(n_components=pca_components)
X_test_pca = pca.fit_transform(X_test_reshaped)

# Concatenate signal origin data, and indices with PCA output
X_test_combined = np.concatenate((X_test_pca, sig_orig_test, test_idx), axis=1)
X_test_combined = X_test_combined.reshape(X_test_combined.shape[0], X_test_combined.shape[1], 1)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(np.argmax(Y_test, axis=1), dtype=torch.long).to(device)

# Load the trained model
model_folder = r'C:\Users\lolaf\Desktop\Thesis_code\model_lr0.001_bs128_pca70.pth'  # Update this path to your model folder
model_path = os.path.join(model_folder)

input_shape = (X_test_combined.shape[1], 1)
model = build_model(input_shape, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Placeholder lists for true labels and scores
y_true = []
y_scores = []

# Run the model on test data and collect predictions
with torch.no_grad():
    for i in range(0, X_test_tensor.size()[0], 64):  # Assuming batch size of 64
        batch_x = X_test_tensor[i:i + 64]
        batch_y = Y_test_tensor[i:i + 64]

        outputs = model(batch_x)

        # Collect true labels and predicted probabilities for Precision-Recall curve
        y_true.extend(batch_y.cpu().numpy())
        y_scores.extend(outputs.softmax(dim=1).cpu().numpy())

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Binarize the output labels for multi-class precision-recall curve
y_true_bin = label_binarize(y_true, classes=range(num_classes))

# Plot precision-recall curve
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
    average_precision = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {class_name} (AP = {average_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plot_filename = 'precision_recall_curve_from_model.png'
plt.savefig(plot_filename)
print(f'Saved plot: {plot_filename}')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true, np.argmax(y_scores, axis=1))

# Save the confusion matrix
with open('confusion_matrix.pkl', 'wb') as f:
    pickle.dump(cm, f)

# Save y_true and y_scores for future use
with open('y_true.pkl', 'wb') as f:
    pickle.dump(y_true, f)

with open('y_scores.pkl', 'wb') as f:
    pickle.dump(y_scores, f)

# Plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plot_filename = 'confusion_matrix.png'
    plt.savefig(plot_filename)
    print(f'Saved plot: {plot_filename}')
    plt.close()

# Plot the confusion matrix
plot_confusion_matrix(cm)

# If you have history data, you can also plot accuracy/loss and metrics
history_filename = 'history.pkl'
if os.path.exists(history_filename):
    with open(history_filename, 'rb') as f:
        history = pickle.load(f)

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
        plt.savefig(plot_filename)
        print(f'Saved plot: {plot_filename}')
        plt.close()

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
        plt.savefig(plot_filename)
        print(f'Saved plot: {plot_filename}')
        plt.close()

    # Plot accuracy and loss
    plot_accuracy_loss(history)

    # Plot F1 score, precision, and recall
    plot_metrics(history)
