import torch
import torch.nn.functional as F


class Second_model(torch.nn.Module):
    def __init__(self, inputSize, outputSize):

        '''
        Args:
            inputSize: size of the input ( embedding )
            outputSize: 2 ( the number of classes )
        '''

        hiddenSize_1 = 256
        hiddenSize_2 = 64

        super(Second_model, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize_1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=int(hiddenSize_1 / 2))
        # self.fc2 = torch.nn.Linear(hiddenSize_1, hiddenSize_2)
        self.fc2 = torch.nn.Linear(int(hiddenSize_1 / 2), hiddenSize_2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=int(hiddenSize_2 / 2))
        # self.fc3 = torch.nn.Linear(hiddenSize_2, outputSize)
        self.fc3 = torch.nn.Linear(int(hiddenSize_2 / 2), outputSize)
        self.bn3 = torch.nn.BatchNorm1d(num_features=outputSize)
        self.dropout = torch.nn.Dropout(0.01)
        self.maxPooling = torch.nn.MaxPool1d(2)

        # for layer in [self.fc1, self.fc2, self.fc3]:
        #     layer.weight.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        '''
        :param x: input to the model
        :return: prediction of size of the number of classes
        '''
        out = self.fc1(x.float())
        out = self.maxPooling(out.unsqueeze(dim=1))
        out = self.bn1(out.squeeze(dim=1))
        out = self.dropout(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = self.maxPooling(out.unsqueeze(dim=1))
        out = self.bn2(out.squeeze(dim=1))
        # out = self.dropout(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)

        out = F.softmax(out, dim=1)
        return out
