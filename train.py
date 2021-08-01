# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import pandas as pd
from torch.autograd import Variable

from datetime import timedelta

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, get_outlier_loader
from utils.dist_util import get_world_size
from utils.file_utils import CSV_Writer

from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
train_csv_writer = None
val_csv_writer = None


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

""""
function for show image and adversarial image--> we hava an error in assignment to model!
def get_attention_map(img, get_mask=False):
    x = transform(img)
    x.size()

    logits, att_mat = model(x.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:        
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")
    
    return result

def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
""""

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    # num_classes = 10 if args.dataset == "cifar10" else 100
    num_classes = 5

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, vis=True)
    if args.pretrained_dir: # if false train from scratch
        model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, epoch, is_normal=True):
    # Validation!
    eval_losses = AverageMeter()
    att_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    all_attn_loss = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X) (att_loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0],
                          position=0, leave=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    att_criterion = torch.nn.MSELoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        noised_x = pgd_attack(x, model, eps=args.fgsm_eps, n_iter=args.fgsm_iter)
        model.zero_grad()
        model.eval()
        with torch.no_grad():
            logits, attn_weights = model(x)
            attn_weights = torch.stack(attn_weights, dim=1)

            _, noisy_attn = model(noised_x)
            noisy_attn = torch.stack(noisy_attn, dim=1)

            attn_loss = att_criterion(attn_weights, noisy_attn)

            att_losses.update(attn_loss.item())
            all_attn_loss.append(attn_loss.item())
            if is_normal:
                eval_loss = loss_fct(logits, y)
                eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f) (att_loss=%2.6f)" % (eval_losses.avg, att_losses.avg))

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Epoch: %d" % epoch)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Attention Loss: %2.6f" % att_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=epoch)
    if is_normal:
        return accuracy, all_attn_loss, eval_losses.avg
    else:
        return accuracy, all_attn_loss


def make_noise(x, epsilon=0.03):
    delta = torch.zeros_like(x).cuda()
    delta.uniform_(-epsilon, epsilon)
    return torch.clamp(delta + x, -1, 1)


def pgd_attack(x, model, eps=0.03, n_iter=10):
    model.eval()
    adv_x = x.detach().clone()
    adv_x = make_noise(adv_x, eps)
    _, target_attn = model(x)
    target_attn = torch.stack(target_attn, dim=1).data
    for i in range(n_iter):
        model.zero_grad()

        adv_x.requires_grad = True
        out, attn = model(adv_x)
        attn = torch.stack(attn, dim=1)
        loss = torch.nn.functional.mse_loss(attn, target_attn)
        loss.backward()
        adv_x.requires_grad = False

        adv_x = adv_x + (2.5/n_iter) * eps * adv_x.grad.sign()
        eta = torch.clamp(adv_x - x, -eps, eps)
        adv_x = torch.clamp(x + eta, -1, 1)
    return adv_x


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    normal_train_loader, normal_test_loader, outlier_train_loader, outlier_test_loader = get_outlier_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    attentions = AverageMeter()
    epoch, best_acc = 0, 0
    att_criterion = torch.nn.MSELoss()
    while True:
        model.train()
        epoch_iterator = tqdm(normal_train_loader,
                              desc="Training (X / X Steps) (loss=X.X) (att_loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0],
                              position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss, attn_weights = model(x, y)
            attn_weights = torch.stack(attn_weights, dim=1)
            noised_x = pgd_attack(x, model, eps=args.pgd_eps, n_iter=args.pgd_iter)
            model.train()
            _, noisy_attn = model(noised_x, y)
            noisy_attn = torch.stack(noisy_attn, dim=1)
            attn_loss = att_criterion(attn_weights, noisy_attn)
            loss += args.attn_loss_coef * attn_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            losses.update(loss.item() * args.gradient_accumulation_steps)
            attentions.update(attn_loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f) (att_loss=%2.6f)" % (epoch, t_total, losses.avg, attentions.avg)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=epoch)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=epoch)

        optimizer.zero_grad()
        if args.local_rank in [-1, 0]:
            epoch += 1
            accuracy, normal_attn_losses, normal_eval_loss = valid(args, model, writer, normal_test_loader, epoch)
            outlier, outlier_attn_losses = valid(args, model, writer, outlier_test_loader, epoch, is_normal=False)
            auc = roc_auc_score([0]*len(normal_attn_losses) + [1]*len(outlier_attn_losses), normal_attn_losses + outlier_attn_losses)
            normal_np = np.array(normal_attn_losses)
            outlier_np = np.array(outlier_attn_losses)
            val_csv_writer.add_record([epoch, auc, normal_np.mean(), outlier_np.mean(),
                                       normal_eval_loss + args.attn_loss_coef * normal_np.mean(), accuracy])
            logger.info('AUC Score: %.5f' % auc)
            if best_acc < accuracy:
                save_model(args, model)
                best_acc = accuracy
            model.train()
            train_csv_writer.add_record([epoch, losses.avg, attentions.avg])
            losses.reset()
            attentions.reset()
        if epoch % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--csv_path", default='./Results/',
                        help="path to store csv result.")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100", "MNIST"], default="MNIST",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, # default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--attn_loss_coef', type=float, default=1e3,
                        help="coefficient of attn_loss when summing two losses")

    parser.add_argument("--pgd_iter", type=int, default=10,
                        help="number of iterations in pgd")
    parser.add_argument("--pgd_eps", type=float, default=0.03,
                        help="epsilon in pgd")


    args = parser.parse_args()

    global train_csv_writer
    global val_csv_writer
    train_csv_writer = CSV_Writer(path=args.csv_path + args.name + '_train.csv',
                                  columns=(['epoch', 'train_loss', 'train_attention_loss']))
    val_csv_writer = CSV_Writer(path=args.csv_path + args.name + '_val.csv',
                                columns=(['epoch', 'auc', 'normal_attention_loss',
                                          'oulier_attention_loss', 'normal_loss', 'accuracy']))

    # Setup CUDA, GPU & distributed training 
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
