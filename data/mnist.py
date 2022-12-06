import torchvision.transforms as transforms
import data

from torchvision.datasets import MNIST, EMNIST
import torch 

def generate_mnist_datasets(BATCH_SIZE, transform, download=False): 
    DATA_ROOT = '../mnist'
    train_dataset = MNIST(
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

    test_dataset = MNIST(
        root=DATA_ROOT, train=False, download=download, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
        
    return train_loader, val_loader, test_loader


def generate_emnist_datasets(BATCH_SIZE, transform): 
    DATA_ROOT = '../emnist'
    train_dataset = EMNIST(
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

    test_dataset = EMNIST(
        root=DATA_ROOT, train=False, download=download, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
        
    return train_loader, val_loader, test_loader


# import os, sys
# import glob
# import time
# import random
# import pickle
# import numpy as np
# import types
# from PIL import Image

# import torch as tc
# from torch.utils.data import DataLoader, Subset
# import torchvision.datasets as datasets

# class MNISTDataset:
#     def __init__(self, x, y, transform):
#         self.x = x
#         self.y = y
#         self.transform = transform

#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, index):
#         sample, target = self.x[index], self.y[index]
#         sample = Image.fromarray(sample)

#         if self.transform is not None:
#             sample, target = self.transform((sample, target))
#         return sample, target
        

# class MNIST:
#     def __init__(self, args):

#         root = os.path.join('data', args.src.lower())
        
#         ## default transforms
#         tforms_dft = [
#             transforms.Grayscale(3),
#             transforms.ToTensor(),
#         ]
        

#         ## transformations for each data split
#         tforms_train = tforms_dft
#         tforms_val = tforms_dft
#         tforms_test = tforms_dft
#         print("[tforms_train] ", tforms_train)
#         print("[tforms_val] ", tforms_val)
#         print("[tforms_test] ", tforms_test)


#         ## load data using pytorch datasets
#         train_ds = datasets.MNIST(root=root, train=True, download=True, transform=None)
#         test_ds = datasets.MNIST(root=root, train=False, download=True, transform=None)

#         ## get splits
#         x_test, y_test = np.array(test_ds.data), np.array(test_ds.targets)
        
#         index_rnd = data.get_random_index(len(x_test), len(x_test), args.seed)
#         index_val = index_rnd[:len(index_rnd)//2] # split by half
#         index_test = index_rnd[len(index_rnd)//2:]

#         x_train, y_train = np.array(train_ds.data), np.array(train_ds.targets)
#         x_val, y_val = x_test[index_val], y_test[index_val]
#         x_test, y_test = x_test[index_test], y_test[index_test]
               
#         ## get class name
#         classes, class_to_idx = train_ds.classes, train_ds.class_to_idx

#         ## create a data loader for training
#         ds = Subset(MNISTDataset(x_train, y_train, classes, class_to_idx, transform=tforms.Compose(tforms_train)),
#                     data.get_random_index(len(y_train), len(y_train), args.seed))
#         self.train = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

#         ## create a data loader for validation
#         ds = Subset(MNISTDataset(x_val, y_val, classes, class_to_idx, transform=tforms.Compose(tforms_val)),
#                     data.get_random_index(len(y_val), len(y_val), args.seed))
#         self.val = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

#         ## create a data loader for test
#         ds = Subset(MNISTDataset(x_test, y_test, classes, class_to_idx, transform=tforms.Compose(tforms_test)),
#                     data.get_random_index(len(y_test), len(y_test), args.seed))
#         self.test = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
        
#         ## print data statistics
#         print(f'#train = {len(self.train.dataset)}, #val = {len(self.val.dataset)}, #test = {len(self.test.dataset)}')

        
# if __name__ == '__main__':
#     dsld = data.MNIST(types.SimpleNamespace(src='MNIST', batch_size=100, seed=0, n_workers=10))


