"""
Main file for training a Convolutional Neural Network for image classification.
Training is performed on CIFAR-10
"""

from __future__ import print_function

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import ConvNet
from models import myConvNet
from models import crossentropy_loss
from utils import progress_bar

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_cifar_loaders():
    """
    Provides dataloaders for CIFAR-10.

    Returns
    -------
    tuple
        trainloader: torch.utils.data.DataLoader
            the DataLoader for training examples
        testloader: torch.utils.data.DataLoader
            the DataLoader for test examples
    """

    # Transforms
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Initialize datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Initialize loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


def train(model, loader, opt):
    """
    Training function. Trains a model for one epoch.

    Parameters
    ----------
    model: torch.nn.Module
        the model to be trained.
    loader: torch.utils.data.DataLoader
        the data loader providing training examples.
    opt: torch.optim.Optimizer
        the optimizer to be employed for training.

    Returns
    -------
    None
    """

    # Set the model in train mode
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    # Loop over training examples
    for batch_idx, (inputs, targets) in enumerate(loader):

        # Get a batch of examples
        inputs, targets = inputs.to(device), targets.to(device)

        # Predict
        outputs = model(inputs)

        # Compute loss
        loss = crossentropy_loss(y_true=targets, y_pred=outputs)

        # Backward pass and update
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Update statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print
        progress_bar('TRAINING', batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(model, loader):
    """
    Test function. Tests a model on the test set.

    Parameters
    ----------
    model: torch.nn.Module
        the model to be tested.
    loader: torch.utils.data.DataLoader
        the data loader providing test examples.

    Returns
    -------
    None
    """

    # Set the model in test mode
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            # Get a batch of examples
            inputs, targets = inputs.to(device), targets.to(device)

            # Predict
            outputs = model(inputs)

            # Compute loss
            loss = crossentropy_loss(y_true=targets, y_pred=outputs)
            test_loss += loss.item()

            # Update statistics
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print
            progress_bar('TESTING', batch_idx, len(loader), 'Loss: %.3f │ Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def main():
    """ The main function. """

    # Get data loaders
    trainloader, testloader = get_cifar_loaders()

    # Get model
    print('==> Building model..')
    model = myConvNet(n_classes=10).to(device)

    # Get optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(0, 10):

        print('\n\nEpoch: %d' % epoch)

        train(model, trainloader, optimizer)
        test(model, testloader)


# entry point
if __name__ == '__main__':
    main()
