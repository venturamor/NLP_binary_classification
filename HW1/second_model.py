import torch
import torch.nn.functional as F


class Second_model(torch.nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize = 300):
        '''
        Args:
            inputSize: vector_size
            outputSize: 2
        '''
        super(Second_model, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize)
        self.fc2 = torch.nn.Linear(hiddenSize, outputSize)
        self.fc2 = torch.nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        out = self.fc1(x.float())
        out = F.relu(out)
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(out)
        return out
