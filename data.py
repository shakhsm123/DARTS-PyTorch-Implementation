import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from utils import Cutout

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD  = [0.24703233, 0.24348505, 0.26158768]


def get_search_loaders(batch_size=64, num_workers=0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    n = len(full_train)
    train_data, val_data = random_split(full_train, [n // 2, n // 2])
    val_data.dataset.transform = val_transform
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    return train_loader, val_loader


def get_eval_loaders(batch_size=96, num_workers=0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(length=16),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_data = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    return train_loader, test_loader
