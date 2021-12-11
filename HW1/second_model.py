import torch
import torch.nn.functional as F


class Second_model(torch.nn.Module):
    def __init__(self, inputSize, outputSize):

        '''
        Args:
            inputSize: vector_size
            outputSize: 2
        '''

        hiddenSize_1 = 64
        hiddenSize_2 = 32

        super(Second_model, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize_1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hiddenSize_1)
        self.fc2 = torch.nn.Linear(hiddenSize_1, hiddenSize_2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hiddenSize_2)
        self.fc3 = torch.nn.Linear(hiddenSize_2, outputSize)
        self.bn3 = torch.nn.BatchNorm1d(num_features=outputSize)
        self.dropout = torch.nn.Dropout(0.2)

        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.weight.data.uniform_(-1, 1)

    def forward(self, x):
        out = self.fc1(x.float())
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = F.softmax(out, dim=1)
        return out

    # def init_weights(m):
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.xavier_uniform(m.weight)
    #         m.bias.data.fill_(0.01)