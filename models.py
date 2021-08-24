import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.fc = nn.Linear(input_size, output_size, bias=False)
            
    def forward(self, x):
        return self.fc(x.reshape(-1, self.input_size)).squeeze()

    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
            
    def forward(self, x):
        out = self.fc1(x.reshape(-1, self.input_size))
        out = self.act(out)
        out = self.fc2(out)
        return out.squeeze()

    
class ConvNet(nn.Module):
    def __init__(self, num_filters, output_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, 5, 1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, 1)
        self.fc1 = nn.Linear(4*4*num_filters, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x.reshape(-1, 1, 28, 28))
        out = self.act(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = self.act(out)
        out = F.max_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return out.squeeze()
    