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
        hiddenSize_2 = 128
        hiddenSize_3 = 64
        hiddenSize_4 = 8

        super(Second_model, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize_1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hiddenSize_1)

        self.fc2 = torch.nn.Linear(hiddenSize_1, hiddenSize_2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hiddenSize_2)

        self.fc3 = torch.nn.Linear(hiddenSize_2, hiddenSize_3)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hiddenSize_3)

        self.fc4 = torch.nn.Linear(hiddenSize_3, hiddenSize_4)
        self.bn4 = torch.nn.BatchNorm1d(num_features=hiddenSize_4)

        self.fc5 = torch.nn.Linear(hiddenSize_4, outputSize)
        self.bn5 = torch.nn.BatchNorm1d(num_features=outputSize)

        self.dropout = torch.nn.Dropout(0.01)

        # for layer in [self.fc1, self.fc2, self.fc3]:
        #     layer.weight.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        '''
        :param x: input to the model
        :return: prediction of size of the number of classes
        '''
        out = self.fc1(x.float())
        out = self.bn1(out)
        # out = self.dropout(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = F.relu(out)


        out = self.fc3(out)
        out = self.bn3(out)
        # out = self.dropout(out)
        out = F.relu(out)

        out = self.fc4(out)
        out = self.bn4(out)
        # out = self.dropout(out)
        out = F.relu(out)

        out = self.fc5(out)
        out = self.bn5(out)

        out = F.softmax(out, dim=1)
        return out
