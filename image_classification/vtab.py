from torchvision import transforms
import numpy as np

_DATASET_NAME = (
    'cifar',
    'caltech101',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'svhn',
    'sun397',
    'patch_camelyon',
    'eurosat',
    'resisc45',
    'diabetic_retinopathy',
    'clevr_count',
    'clevr_dist',
    'dmlab',
    'kitti',
    'dsprites_loc',
    'dsprites_ori',
    'smallnorb_azi',
    'smallnorb_ele',
)
_CLASSES_NUM = (10, 102, 47, 102, 37, 10, 397, 2, 10, 45, 5, 8, 6, 6, 4, 16, 16, 18, 9)

def get_classes_num(dataset_name):
    dict_ = {name: num for name, num in zip(_DATASET_NAME, _CLASSES_NUM)}
    return dict_[dataset_name]

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Step 3: Custom Dataset Class
class subset_wrapper(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

# Function containing Steps 1, 2, and 4
def get_data(name, evaluate=True, batch_size=64, train_samples_num=800, seed=27):
    # Specified Transform
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((224, 224), interpolation=3), # ViTB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 1: Download
    # Assumes dataset is in torchvision.datasets (e.g., 'CIFAR10'
    if name == 'caltech101':
        dataset_class = getattr(datasets, 'Caltech101')
        # Caltech101 does not have a train/test split argument
        raw_data = dataset_class(root='~/.torch/data', download=True)
        # Caltech101 stores labels in .y, not .targets
        targets = np.array(raw_data.y)
    elif name == 'cifar':
        dataset_class = getattr(datasets, 'CIFAR10')
        # Default behavior for CIFAR10/100
        raw_data = dataset_class(root='~/.torch/data', train=True, download=True)
        targets = np.array(raw_data.targets)

    images_per_class = int(train_samples_num/get_classes_num(name))

    # Step 2: Stratified Split (8 images per class for 100 classes = 800)
    train_indices = []
    val_indices = []
    test_indices = []

    classes = np.unique(targets)
    print(f'# of total classes: {len(classes)}') 
    np.random.seed(seed)
    for c in classes:
        # Get all indices for this class
        c_indices = np.where(targets == c)[0]
        # Shuffle them to be safe
        np.random.shuffle(c_indices)
        
        # Take first 8 for train
        train_indices.extend(c_indices[:images_per_class])
        # Take next 2 for val (example)
        val_indices.extend(c_indices[images_per_class:int(images_per_class*1.25)])
        test_indices.extend(c_indices[int(images_per_class*1.25):int(images_per_class*1.5)])  # Remaining for test

    # Instantiate Step 3 Class
    train_set = subset_wrapper(raw_data, train_indices, transform=transform)
    val_set = subset_wrapper(raw_data, val_indices, transform=transform)
    test_set = subset_wrapper(raw_data, test_indices, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if evaluate:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

