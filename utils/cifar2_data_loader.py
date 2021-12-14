"""
-*-coding = utf-8 -*-
__author: topsy
@time:2021/12/12 23:12
"""

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms, datasets
import os

img_size = 224




def get_cifar2_dataset(args = None):
    print("***** Current use the cifar2 dataset for train and test.  *****")
    cifar2_dir = './data/hymenoptera_data'
    if args:
        img_size = args.img_size
    else:
        img_size = 224
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.ImageFolder(root=os.path.join(cifar2_dir, "train"),
                                    transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(cifar2_dir, "val"),
                                   transform=transform_test)
    return trainset, testset


def get_cifar2_dataloader(args = None):
    if args:
        train_batch_size = args.train_batch_size
        eval_batch_size = args.eval_batch_size
    else:
        train_batch_size = 32
        eval_batch_size = 32
    trainset, testset = get_cifar2_dataset()
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


