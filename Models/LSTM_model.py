import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_shape[1], 128, batch_first=True) 
        self.batch_norm1 = nn.BatchNorm1d(128) 

        self.lstm2 = nn.LSTM(128, 64, batch_first=True) 
        self.batch_norm2 = nn.BatchNorm1d(64)
        
        self.fc1 = nn.Linear(64, 32)
        self.dropout_fc1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.batch_norm1(x[:, -1, :])
        
        x, _ = self.lstm2(x.unsqueeze(1))
        x = self.batch_norm2(x[:, -1, :])
        
        x = self.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x

def build_model(input_shape, num_classes):
    return LSTMModel(input_shape, num_classes)

