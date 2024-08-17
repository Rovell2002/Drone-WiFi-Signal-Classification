import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Conv1DModel, self).__init__()
        
        self.relu = nn.ReLU()  # Define ReLU activation here
        
        # Conv1D layers with appropriate adjustments
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(16)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size of the output after Conv1D and Pooling layers
        self.flatten_size = self._get_flatten_size(input_shape)
        
        # Fully connected layer
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def _get_flatten_size(self, input_shape):
        # Use a batch size of 2 instead of 1 to avoid batch norm issues
        x = torch.rand(2, *input_shape)
        x = self.relu(self.pool1(self.bn1(self.conv1(x))))
        x = self.relu(self.pool2(self.bn2(self.conv2(x))))
        x = self.relu(self.pool3(self.bn3(self.conv3(x))))
        x = self.relu(self.pool4(self.bn4(self.conv4(x))))
        return x.view(2, -1).size(1)  # Flatten and get the size
    
    def forward(self, x):
        # Conv1D layers
        x = self.relu(self.pool1(self.bn1(self.conv1(x))))
        x = self.relu(self.pool2(self.bn2(self.conv2(x))))
        x = self.relu(self.pool3(self.bn3(self.conv3(x))))
        x = self.relu(self.pool4(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    
def build_model(input_shape, num_classes):
    return Conv1DModel(input_shape, num_classes)
