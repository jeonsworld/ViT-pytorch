"""
-*-coding = utf-8 -*-
__author: topsy
@time:2021/12/13 21:55
"""

import torch
import logging
 
logger = logging.getLogger(__name__)

from utils.cifar2_data_loader import get_cifar2_dataset, get_cifar2_dataloader
from torch.nn import CrossEntropyLoss
from models.modeling import VisionTransformer, CONFIGS
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def get_model(pretrained_dir, model_type, num_classes=2, img_size=224, ):
    config = CONFIGS[model_type]
    model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
    model.load_state_dict(torch.load(pretrained_dir))
    return model

def valid(model, test_loader, dataset_sizes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Validation!
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", test_loader.batch_size)
    model.eval()
    model.to(device)
    correct_count = 0
    loss = 0.0
    loss_fct = CrossEntropyLoss()
    error_cases = None
    error_cases_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            _, preds = torch.max(logits, dim=-1)
            batch_loss = loss_fct(logits.view(-1, 2), y.view(-1))
            loss += batch_loss.item() * x.size(0)
            correct_count += torch.sum(preds == y.data)
            batch_error_caese = x[preds != y.data]
            if len(batch_error_caese) > 0:
                if error_cases != None:
                    error_cases = torch.vstack([error_cases, batch_error_caese])
                else:
                    error_cases = batch_error_caese
                error_cases_labels += list(preds[preds != y.data])
                

    avg_loss = loss * 1.0 / dataset_sizes["test"]
    avg_acc = correct_count * 1.0 / dataset_sizes["test"]
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
    return avg_loss, avg_acc, error_cases, error_cases_labels



def get_valid_res(pretrained_dir, model_type):
    class_name = {0: "ants", 1: "bees"}
    trainset, testset = get_cifar2_dataset()
    dataset_sizes = {"train": len(trainset), "test": len(testset)}
    train_loader, test_loader = get_cifar2_dataloader()
    model = get_model(pretrained_dir, model_type)
    avg_loss, avg_acc, error_cases, error_cases_labels = valid(model, test_loader, dataset_sizes)
    return error_cases, error_cases_labels



def imshow(error_cases, error_cases_labels):
    class_name = {0: "ants", 1: "bees"}
    img = torchvision.utils.make_grid(error_cases, 4)
    gt_labels  = [class_name[id.detach().cpu().item()][:-1] for id in error_cases_labels]
    title = ", ".join(gt_labels)
    img = img/2 +0.5
    npimg = img.detach().cpu().numpy()
    plt.figure(figsize=(16,16))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
    


