from torchvision.datasets import CIFAR10
import torch 

DATA_ROOT = '../cifar10'

def generate_cifar_datasets(BATCH_SIZE, transform, download=True): 
    train_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=download, transform=transform)
    
    train_len = int(len(train_dataset) * 0.8)
    val_len = len(train_dataset) - train_len
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
                                                               [train_len, val_len], 
                                                               generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_dataset = CIFAR10(
        root=DATA_ROOT, train=False, download=download, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
        
    return train_loader, val_loader, test_loader