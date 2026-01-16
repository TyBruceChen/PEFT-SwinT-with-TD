import torchvision
from torchvision import datasets, models, transforms
import torch

def get_torch_dataset(dataset_name,
                      train_ratio = 0.8,
                      target_size=(224, 224),
                      main_path='~/.cache/torch/hub/checkpoints/'):

  transform = transforms.Compose([
          transforms.Lambda(lambda x: x.convert("RGB")), # Force RGB (handles grayscale issues)
          transforms.Resize(240),  # Oversize first
          transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Sharp random crop
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  if dataset_name == 'cifar10':
    cifar10_trainset = torchvision.datasets.CIFAR10(root = main_path+"cifar10",
                                transform=transform,
                                download=True)
    cifar10_testset = torchvision.datasets.CIFAR10(root = main_path+"cifar10",
                                  train=False,
                                  transform=transform,)
    print(cifar10_trainset.data.shape)
    return cifar10_trainset, cifar10_testset
  elif dataset_name == 'mnist':
    mnist_trainset = torchvision.datasets.MNIST(root = main_path+"MNIST",
                                transform=transform,
                                download=True)
    mnist_testset = torchvision.datasets.MNIST(root = main_path+"MNIST",
                                  train=False,
                                  transform=transform,)
    print(mnist_trainset.data.shape)
    return mnist_trainset, mnist_testset
  elif dataset_name == 'caltech101':
    caltech101_trainset = torchvision.datasets.Caltech101(root = main_path+"caltech101",
                                transform=transform,
                                download=True)
    full_size = len(caltech101_trainset)
    train_size = int(train_ratio*full_size)
    print(f"Raw size: {full_size}")
    caltech101_trainset, caltech101_testset = torch.utils.data.random_split(caltech101_trainset,
                        [train_size, full_size - train_size])
    print(f"trainset: {len(caltech101_trainset)}, testset: {len(caltech101_testset)}")
    return caltech101_trainset, caltech101_testset