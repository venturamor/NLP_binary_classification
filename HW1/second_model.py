import torch
import torch.nn.functional as F


class Second_model(torch.nn.Module):
    def __init__(self, inputSize, outputSize):

        '''
        Args:
            inputSize: vector_size
            outputSize: 2
        '''

        hiddenSize_1 = 256
        hiddenSize_2 = 256

        super(Second_model, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize_1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hiddenSize_1)
        self.fc2 = torch.nn.Linear(hiddenSize_1, hiddenSize_2)
        # self.fc2 = torch.nn.Linear(int(hiddenSize_1 / 2), hiddenSize_2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hiddenSize_2)
        self.fc3 = torch.nn.Linear(hiddenSize_2, outputSize)
        # self.fc3 = torch.nn.Linear(int(hiddenSize_2 / 2), outputSize)
        self.bn3 = torch.nn.BatchNorm1d(num_features=outputSize)
        self.dropout = torch.nn.Dropout(0.01)
        # self.maxPooling = torch.nn.MaxPool1d(3)

    def forward(self, x):
        out = self.fc1(x.float())
        # out = self.maxPooling(out)
        out = self.bn1(out)
        out = self.dropout(out)
        out = F.relu(out)

        out = self.fc2(out)
        # out = self.maxPooling(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = F.softmax(out, dim=1)
        return out
