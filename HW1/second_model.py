import torch
import torch.nn.functional as F


class Second_model(torch.nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize=300):

        '''
        Args:
            inputSize: vector_size
            outputSize: 2
        '''

        super(Second_model, self).__init__()
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize)
        self.fc2 = torch.nn.Linear(hiddenSize, outputSize)
        self.fc2 = torch.nn.Linear(hiddenSize, outputSize)

        net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        net.apply(self.init_weights)

    def forward(self, x):
        out = self.fc1(x.float())
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.softmax(out, dim=1)
        return out

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)