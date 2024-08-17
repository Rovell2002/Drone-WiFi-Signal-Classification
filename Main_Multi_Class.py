import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from LSTM_model import build_model # Change the models here for running them
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve
import pickle
from torchsummary import summary

# Ensure PyTorch uses the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for training")

# Define classes and parameters
classes = ["dx4e", "dx6i", "MTx", "Nineeg", "Parrot", "q205", "S500", "tello", "WiFi", "wltoys"] #Multi_class Classifcation

num_epochs = 100
lr = 0.01
batch_size = 128
pca_components = 70
num_classes = len(classes)

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
Y_train = np.array(h5f['Y_train'])
X_test = np.array(h5f['X_test'])
Y_test = np.array(h5f['Y_test'])
h5f.close()
h5f1.close()
h5f2.close()

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

scaler_sig_orig = StandardScaler()
sig_orig_train = scaler_sig_orig.fit_transform(sig_orig_train)
sig_orig_test = scaler_sig_orig.transform(sig_orig_test)

scaler_idx = StandardScaler()
train_idx = scaler_idx.fit_transform(train_idx)
test_idx = scaler_idx.transform(test_idx)

def run_experiment(lr, batch_size, pca_components):
    
    # Reshape and apply PCA
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_reshaped)
    X_test_pca = pca.transform(X_test_reshaped) 

    # Concatenate signal origin data, and indices with PCA output
    X_train_combined = np.concatenate((X_train_pca, sig_orig_train, train_idx), axis=1)
    X_test_combined = np.concatenate((X_test_pca, sig_orig_test, test_idx), axis=1)

    # Reshape combined data to fit the input shape expected by the model
    X_train_combined = X_train_combined.reshape(X_train_combined.shape[0], X_train_combined.shape[1], 1)
    X_test_combined = X_test_combined.reshape(X_test_combined.shape[0], X_test_combined.shape[1], 1)

    input_shape = (X_train_combined.shape[1], 1)

    # Initialize and compile the model
    model = build_model(input_shape, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(np.argmax(Y_test, axis=1), dtype=torch.long).to(device)

    # Train the model
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [], 'f1_score': [], 'precision': [], 'recall': []}

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])

        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], Y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_accuracy = 100 * correct / total
        history['train_loss'].append(epoch_loss / total)
        history['train_accuracy'].append(epoch_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        y_true = []
        y_scores = []

        with torch.no_grad():
            for i in range(0, X_test_tensor.size()[0], batch_size):
                batch_x, batch_y = X_test_tensor[i:i + batch_size], Y_test_tensor[i:i + batch_size]
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
                # Collect true labels and predicted probabilities for Precision-Recall curve
                y_true.extend(batch_y.cpu().numpy())
                y_scores.extend(outputs.softmax(dim=1).cpu().numpy()[:, 1]) 

        val_accuracy = 100 * val_correct / val_total
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        history['val_loss'].append(val_loss / val_total)
        history['val_accuracy'].append(val_accuracy)
        history['f1_score'].append(f1)
        history['precision'].append(precision)
        history['recall'].append(recall)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / total:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, Val Loss: {val_loss / val_total:.4f}, Val Accuracy: {val_accuracy:.2f}%, F1 Score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    return history, model, cm, y_true, y_scores

# Run the experiment
history, model, cm, y_true, y_scores = run_experiment(lr, batch_size, pca_components)

# Save the results
with open('history.pkl', 'wb') as f:
    pickle.dump(history, f)

torch.save(model.state_dict(), 'model.pth')

with open('confusion_matrix.pkl', 'wb') as f:
    pickle.dump(cm, f)

# Save y_true and y_scores for Precision-Recall curve
with open('y_true.pkl', 'wb') as f:
    pickle.dump(y_true, f)

with open('y_scores.pkl', 'wb') as f:
    pickle.dump(y_scores, f)
