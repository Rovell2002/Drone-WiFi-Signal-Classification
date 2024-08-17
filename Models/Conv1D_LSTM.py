import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DLSTMModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Conv1DLSTMModel, self).__init__()
        
        # Conv1D layers
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.AvgPool1d(kernel_size=1)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.AvgPool1d(kernel_size=1)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.AvgPool1d(kernel_size=1)
        
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size of the output after Conv1D and Pooling layers
        self.lstm_input_size = 32  # Since we are reducing features
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=self.lstm_input_size, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print("Input shape:", x.shape)
        # Conv1D layers
        x = self.relu(self.pool1(self.bn1(self.conv1(x))))
        print("After pool1:", x.shape)
        x = self.relu(self.pool2(self.bn2(self.conv2(x))))
        print("After pool2:", x.shape)
        x = self.relu(self.pool3(self.bn3(self.conv3(x))))
        print("After pool3:", x.shape)
        x = self.dropout(x)
        
        # Transpose to match LSTM input shape (batch_size, seq_len, feature_size)
        x = x.transpose(1, 2)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Take the last output of the sequence
        x = x[:, -1, :]
        x = self.dropout(x)
        
        # Output layer
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x
    
def build_model(input_shape, num_classes):
    return Conv1DLSTMModel(input_shape, num_classes)