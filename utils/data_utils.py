import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils.cifar_outlier_dataset import CifarOutlierDataset

logger = logging.getLogger(__name__)


def get_cifar_outlier_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    normal_trainset = CifarOutlierDataset(args, transform_train, train=True, is_normal=True)
    normal_testset = CifarOutlierDataset(args, transform_test, train=False, is_normal=True)
    outlier_trainset = CifarOutlierDataset(args, transform_train, train=True, is_normal=False)
    outlier_testset = CifarOutlierDataset(args, transform_test, train=False, is_normal=False)

    if args.local_rank == 0:
        torch.distributed.barrier()

    normal_train_sampler = RandomSampler(normal_trainset) if args.local_rank == -1 else DistributedSampler(normal_trainset)
    normal_test_sampler = SequentialSampler(normal_testset)
    normal_train_loader = DataLoader(normal_trainset,
                              sampler=normal_train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    normal_test_loader = DataLoader(normal_testset,
                             sampler=normal_test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if normal_testset is not None else None

    outlier_train_sampler = RandomSampler(outlier_trainset) if args.local_rank == -1 else DistributedSampler(outlier_trainset)
    outlier_test_sampler = SequentialSampler(outlier_testset)
    outlier_train_loader = DataLoader(outlier_trainset,
                              sampler=outlier_train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    outlier_test_loader = DataLoader(outlier_testset,
                             sampler=outlier_test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if outlier_testset is not None else None

    return normal_train_loader, normal_test_loader, outlier_train_loader, outlier_test_loader


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
