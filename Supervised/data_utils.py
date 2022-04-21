import torchvision
import torchvision.transforms as transforms
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat

means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_imagenet_loaders(root_path="data/imagenette2", **kwargs):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )

    trainset = torchvision.datasets.ImageFolder(
        os.path.join(root_path, "train"), transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, **kwargs)
    testset = torchvision.datasets.ImageFolder(
        os.path.join(root_path, "val"), transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(testset, **kwargs)
    return trainloader, testloader


mnist_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)


def get_mnist_usps_loaders(root_path="./data", transform=mnist_transform, **kwargs):

    trainset = torchvision.datasets.MNIST(
        root=root_path, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.USPS(
        root=root_path, train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, **kwargs)
    return trainloader, testloader


def get_stl10_loaders(root_path="./data", **kwargs):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )
    trainset = torchvision.datasets.STL10(
        root=root_path, split="train", download=True, transform=train_transform
    )
    testset = torchvision.datasets.STL10(
        root=root_path, split="test", download=True, transform=test_transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, **kwargs)
    return trainloader, testloader
