from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models import Linear, MLP, ConvNet
import numpy as np
import math
import random
import sys
import utils
from opacus.autograd_grad_sample import add_hooks, remove_hooks, enable_hooks, disable_hooks


def attack_gradient(grad, attack_method='none'):
    '''
    Implement attacks here
    '''
    return grad


def gradient_aggregator(grad, defense_method='none'):
    '''
    Implement defenses here
    '''
    return grad.mean(0)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # delete both gradient and per-sample gradient
        optimizer.zero_grad()
        for param in model.parameters():
            if hasattr(param, 'grad_sample'):
                del param.grad_sample
            
        # enable hook to start computing per-example gradient
        enable_hooks()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        disable_hooks()
        
        # retrieve per-sample gradient and simulate attack/defense
        grad = utils.params_to_vec(model, return_type='grad_sample')
        grad = attack_gradient(grad)
        grad = gradient_aggregator(grad)
        utils.set_grad_to_vec(model, grad)
        
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    return train_loss / len(train_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            correct += output.max(1)[1].eq(target).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='training data directory')
    parser.add_argument('--output-dir', type=str, default='checkpoint/',
                        help='data directory for saving models')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='which dataset to train on')
    parser.add_argument('--model', type=str, default='linear', metavar='M',
                        help='model type (default: linear)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                        help='learning rate (default: 1e-1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    X_train, y_train = utils.load_dataset(args.dataset, args.data_dir, train=True)
    X_test, y_test = utils.load_dataset(args.dataset, args.data_dir, train=False)
        
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    testset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **kwargs)

    output_size = int(y_train.max()) + 1
    if args.model == 'mlp':
        model = MLP(784, 500, output_size).to(device)
    elif args.model == 'cnn':
        model = ConvNet(16, output_size).to(device)
    elif args.model == 'large_cnn':
        model = ConvNet(256, output_size).to(device)
    elif args.model == 'linear':
        model = Linear(784, output_size).to(device)
    else:
        print('Error: Unknown model %s' % args.model)
        sys.exit(1)
    add_hooks(model)
    # the hook will append new per-example gradients to param.grad_sample for each param
    # it's best to disable the hook when per-example gradient is not required
    disable_hooks()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
            
    best_model = model.state_dict()
    best_train = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        # store best model according to training loss
        if train_loss < best_train:
            best_train = train_loss
            best_model = model.state_dict()
            print('Saving...')
        # drop learning rate at half-way point
        if epoch == int(args.epochs / 2):
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
            
    if args.save_model:
        torch.save(best_model, "%s/%s_%s.pth" % (args.output_dir, args.dataset, args.model))


if __name__ == '__main__':
    main()