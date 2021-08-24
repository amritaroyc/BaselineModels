import torch
from torchvision import datasets, transforms
import sys


def load_dataset(dataset, data_dir, train=False):
    if dataset == 'mnist':
        dset = datasets.MNIST(data_dir, train=train, download=True, transform=transforms.ToTensor())
    elif dataset == 'fmnist':
        dset = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transforms.ToTensor())
    elif dataset == 'kmnist':
        dset = datasets.KMNIST(data_dir, train=train, download=True, transform=transforms.ToTensor())
    elif dataset == 'emnist':
        dset = datasets.EMNIST(data_dir, split='digits', train=train, download=True, transform=transforms.ToTensor())
    elif dataset == 'emnist-letters':
        dset = datasets.EMNIST(data_dir, split='letters', train=train, download=True, transform=transforms.ToTensor())
    else:
        print('Error: Unknown dataset %s' % dataset)
        sys.exit(1)
    Xs = torch.zeros(len(dset), 784)
    ys = torch.zeros(len(dset)).long()
    # usually a bad idea to load the data tensor in all at once, but this allows easier assignment of data to clients
    for i in range(len(dset)):
        x, y = dset[i]
        Xs[i] = x.view(784) - 0.5
        ys[i] = y
    return Xs, ys


def params_to_vec(model, return_type='param'):
    '''
    Helper function that concatenates model parameters or gradients into a single vector.
    '''
    vec = []
    for param in model.parameters():
        if return_type == 'param':
            vec.append(param.data.view(1, -1))
        elif return_type == 'grad':
            vec.append(param.grad.view(1, -1))
        elif return_type == 'grad_sample':
            if hasattr(param, 'grad_sample'):
                vec.append(param.grad_sample.view(param.grad_sample.size(0), -1))
            else:
                print('Error: Per-sample gradient not found')
                sys.exit(1)
    return torch.cat(vec, dim=1).squeeze()


def set_grad_to_vec(model, vec):
    '''
    Helper function that sets the model's gradient to a given vector.
    '''
    model.zero_grad()
    for param in model.parameters():
        size = param.data.view(1, -1).size(1)
        param.grad = vec[:size].view_as(param.data).clone()
        vec = vec[size:]
    return