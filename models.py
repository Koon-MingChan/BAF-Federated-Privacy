import torch
import torch.nn as nn
import torch.nn.functional as F

class BAFModel(nn.Module):
    def __init__(self, input_dim):
        super(BAFModel, self).__init__()
        # Simple 3-layer MLP
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1) # Binary classification (Fraud or Not)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # We use sigmoid at the end for probability
        #return torch.sigmoid(self.fc3(x))
        # We now return the raw "logits" because BCEWithLogitsLoss is more stable.
        return self.fc3(x)

def get_model(input_dim):
    return BAFModel(input_dim)