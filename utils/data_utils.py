import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.normal_dataset import NormalDataset
from utils.outlier_dataset import OutlierDataset
from torch.utils.data import random_split

from utils.test_dataset import TestDataset

logger = logging.getLogger(__name__)


def get_transformers(args):
    if args.dataset == "MNIST":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
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
    return transform_train, transform_test


def get_outlier_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train, transform_test = get_transformers(args)
    dataset_method = getattr(datasets, args.dataset)
    train_dataset = dataset_method(root="./data",
                                   train=True,
                                   download=True)
    train_size = int(.8 * len(train_dataset))
    train_set, val_set = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
    normal_trainset = NormalDataset(train_set, transform_train)
    outlier_trainset = OutlierDataset(train_set, transform_train)

    normal_valset = NormalDataset(val_set, transform_test)
    outlier_valset = OutlierDataset(val_set, transform_test)

    normal_testset = TestDataset(args, transform_test, train=False, is_normal=True)
    outlier_testset = TestDataset(args, transform_test, train=False, is_normal=False)

    if args.local_rank == 0:
        torch.distributed.barrier()

    normal_train_sampler = RandomSampler(normal_trainset) if args.local_rank == -1 else DistributedSampler(
        normal_trainset)
    normal_val_sampler = SequentialSampler(normal_valset)
    normal_test_sampler = SequentialSampler(normal_testset)
    normal_train_loader = DataLoader(normal_trainset,
                                     sampler=normal_train_sampler,
                                     batch_size=args.train_batch_size,
                                     num_workers=4,
                                     pin_memory=True)
    normal_val_loader = DataLoader(normal_valset,
                                   sampler=normal_val_sampler,
                                   batch_size=args.eval_batch_size,
                                   num_workers=4,
                                   pin_memory=True) if normal_valset is not None else None
    normal_test_loader = DataLoader(normal_testset,
                                    sampler=normal_test_sampler,
                                    batch_size=args.eval_batch_size,
                                    num_workers=4,
                                    pin_memory=True) if normal_testset is not None else None

    outlier_train_sampler = RandomSampler(outlier_trainset) if args.local_rank == -1 else DistributedSampler(
        outlier_trainset)
    outlier_val_sampler = SequentialSampler(outlier_valset)
    outlier_test_sampler = SequentialSampler(outlier_testset)
    outlier_train_loader = DataLoader(outlier_trainset,
                                      sampler=outlier_train_sampler,
                                      batch_size=args.train_batch_size,
                                      num_workers=4,
                                      pin_memory=True)
    outlier_val_loader = DataLoader(outlier_valset,
                                    sampler=outlier_val_sampler,
                                    batch_size=args.eval_batch_size,
                                    num_workers=4,
                                    pin_memory=True) if outlier_valset is not None else None
    outlier_test_loader = DataLoader(outlier_testset,
                                     sampler=outlier_test_sampler,
                                     batch_size=args.eval_batch_size,
                                     num_workers=4,
                                     pin_memory=True) if outlier_testset is not None else None

    return normal_train_loader, normal_val_loader, normal_test_loader, \
           outlier_train_loader, outlier_val_loader, outlier_test_loader
