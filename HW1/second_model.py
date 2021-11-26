import torch


class Second_model(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        '''
        Args:
            inputSize: vector_size
            outputSize: 2
        '''
        super(Second_model, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
