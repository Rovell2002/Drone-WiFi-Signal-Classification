import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle

import pickle
import seaborn as sns

# Define classes and parameters
# classes = ["dx4e", "dx6i", "MTx", "Nineeg", "Parrot", "q205", "S500", "tello", "WiFi", "wltoys"] # Multi-class Classification
classes = ["drone", "wifi"]  # Binary Classification
num_classes = len(classes)

pca_components = 70

# Load transformed data
filename1 = r'C:\Users\lolaf\Desktop\Thesis_code\Data\Resized_128.h5'
filename2 = r'C:\Users\lolaf\Desktop\Thesis_code\Data\GenDataRSNT_10_snrss2_NW.h5'
filename3 = r'C:\Users\lolaf\Desktop\Thesis_code\Data\binary_test_train_values.h5'
h5f = h5py.File(filename1, 'r')
h5f1 = h5py.File(filename2, 'r')
h5f2 = h5py.File(filename3, 'r')
train_idx = np.array(h5f1['train_idx']).reshape(-1, 1)
test_idx = np.array(h5f1['test_idx']).reshape(-1, 1)
sig_orig_train = np.array(h5f1['sig_origidx_F_train'])
sig_orig_test = np.array(h5f1['sig_origidx_F_test'])
X_train = np.array(h5f['X_train'])
# Y_train = np.array(h5f['Y_train'])
Y_train = np.array(h5f2['Y_train'])  # Binary classification
X_test = np.array(h5f['X_test'])
# Y_test = np.array(h5f['Y_test'])
Y_test = np.array(h5f2['Y_test'])  # Binary classification
h5f.close()
h5f1.close()
h5f2.close()

Y_train = Y_train[:10500]
print("Data loading completed.")

# Convert one-hot encoded labels to class indices if needed
if Y_train.ndim > 1:
    Y_train = np.argmax(Y_train, axis=1)
if Y_test.ndim > 1:
    Y_test = np.argmax(Y_test, axis=1)

print("--" * 10)
print("Training data size:", X_train.shape)
print("Train idx data size:", train_idx.shape)
print("Signal origin Train data size:", sig_orig_train.shape)
print("Training labels size:", Y_train.shape)
print("Testing data size:", X_test.shape)
print("Signal origin Test data size:", sig_orig_test.shape)
print("Test idx data size:", test_idx.shape)
print("Testing labels size:", Y_test.shape)
print("--" * 10)

def run_experiment(X_train, Y_train, X_test, Y_test, pca_components):
    print("Starting experiment...")
    # Reshape and apply PCA
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    print("Applying PCA...")
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_reshaped)
    X_test_pca = pca.transform(X_test_reshaped)

    # Scale the data
    scaler = StandardScaler()
    X_train_pca = scaler.fit_transform(X_train_pca)
    X_test_pca = scaler.transform(X_test_pca)

    # Use only PCA output for training and testing
    print("Using only PCA-transformed features for training and testing...")
    X_train_combined = X_train_pca
    X_test_combined = X_test_pca

    # Initialize the SVM model
    print("Initializing SVM model...")
    model = SVC(probability=True)

    # Train the model
    print("Training the model...")
    model.fit(X_train_combined, Y_train)

    # Validate the model
    print("Validating the model...")
    predictions = model.predict(X_test_combined)
    report = classification_report(Y_test, predictions)
    cm = confusion_matrix(Y_test, predictions)

    # Collect true labels and predicted probabilities for Precision-Recall curve
    y_scores = model.predict_proba(X_test_combined)[:, 1]  # Use probability for the positive class

    f1 = f1_score(Y_test, predictions, average='macro', zero_division=0)
    precision = precision_score(Y_test, predictions, average='macro', zero_division=0)
    recall = recall_score(Y_test, predictions, average='macro', zero_division=0)
    accuracy = np.mean(predictions == Y_test)

    history = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'classification_report': report
    }

    print("Experiment completed.")
    return history, model, cm, Y_test, y_scores

# Run the experiment for binary classification
print("Running the experiment for binary classification...")
history, model, cm, y_true, y_scores = run_experiment(X_train, Y_train, X_test, Y_test, pca_components)

# Save the results for binary classification
print("Saving the results...")
with open('history_binary.pkl', 'wb') as f:
    pickle.dump(history, f)

with open('model_binary.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('confusion_matrix_binary.pkl', 'wb') as f:
    pickle.dump(cm, f)

# Save y_true and y_scores for Precision-Recall curve
with open('y_true_binary.pkl', 'wb') as f:
    pickle.dump(y_true, f)

with open('y_scores_binary.pkl', 'wb') as f:
    pickle.dump(y_scores, f)

print("All results saved successfully.")

# Plotting the results
def plot_results(history, cm, y_true, y_scores, classes):
    # Plot F1 Score, Precision, Recall, and Accuracy
    metrics = ['f1_score', 'precision', 'recall', 'accuracy']
    for metric in metrics:
        plt.figure()
        plt.bar([metric], [history[metric]], label=metric.capitalize())
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('Metrics')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(f'{metric}.png')
        plt.close()

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for binary classification')
    plt.legend(loc="best")
    plt.savefig('precision_recall_curve.png')
    plt.close()

print("Plotting the results...")
plot_results(history, cm, y_true, y_scores, classes)
print("Plots saved successfully.")

