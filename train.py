# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import random
import numpy as np
import pandas as pd
from torch.autograd import Variable
import cv2
from datetime import timedelta
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_outlier_loader
from utils.dist_util import get_world_size
from utils.file_utils import CSV_Writer
from sklearn.metrics import roc_auc_score
from PIL import Image
import os

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


def save_model_training(args, epoch, model, optimizer, scheduler, normal_train_loader):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loader': normal_train_loader
    }
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(state, f"checkpoints/{args.name}_cp.bin")


def load_model_training(args, model, optimizer, scheduler):
    state = torch.load(args.training_weights_dir)
    model.zero_head = False
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    return state['epoch'], state['train_loader']


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    # num_classes = 10 if args.dataset == "cifar10" else 100
    num_classes = 7

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, vis=True)
    if args.pretrained_dir:  # if false train from scratch
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
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def visualize_attention(x, mask, epoch, is_normal, des_name="Pics_atten"):
    x = x.clone()
    mask = mask.clone()
    if not os.path.exists(f"{des_name}"):
        os.makedirs(f"{des_name}")
    if not os.path.exists(f"{des_name}/{epoch}_{is_normal}"):
        os.makedirs(f"{des_name}/{epoch}_{is_normal}")
    for (step, (im, mask_att)) in enumerate(zip(x, mask)):
        mask_att = mask_att.detach().to('cpu').numpy()
        im = im.detach().to('cpu').permute(1, 2, 0).numpy()
        mask_resize = cv2.resize(mask_att / mask_att.max(), im.shape[:2])[..., np.newaxis]
        im = (0.5 * im + 0.5) * 255
        masked_im = (mask_resize * im).astype("uint8")
        heat_map = cv2.applyColorMap((mask_resize[:, :, 0] * 255).astype("uint8"), cv2.COLORMAP_HOT)
        im = transforms.ToPILImage()(im.astype("uint8")).convert("RGB")
        masked_im = transforms.ToPILImage()(masked_im).convert("RGB")
        heat_map = transforms.ToPILImage()(heat_map[:, :, [2, 1, 0]]).convert("RGB")
        image_grid([im, masked_im, heat_map], 1, 3).save(f'{des_name}/{epoch}_{is_normal}/im_{step}.jpg')


def visualize(x, noised_x, epoch, is_normal, des_name="Pics"):
    if not os.path.exists(f"{des_name}"):
        os.makedirs(f"{des_name}")
    if not os.path.exists(f"{des_name}/{epoch}_{is_normal}"):
        os.makedirs(f"{des_name}/{epoch}_{is_normal}")
    for (step, (im, noised_im)) in enumerate(zip(x, noised_x)):
        diff_im = 10 * (im - noised_im)
        diff_im.mul_(0.5).add_(0.5)
        noised_im.mul_(0.5).add_(0.5)
        diff_im = transforms.ToPILImage()(diff_im.to('cpu')).convert("RGB")
        noised_im = transforms.ToPILImage()(noised_im.to('cpu')).convert("RGB")
        image_grid([noised_im, diff_im], 1, 2).save(f'{des_name}/{epoch}_{is_normal}/im_{step}.jpg')


def get_att_mask(att_mat):
    att_mat = torch.stack(att_mat).detach()
    att_mat = torch.mean(att_mat, dim=2)
    residual_att = torch.eye(att_mat.size(2)).to('cuda')
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size()).to('cuda')
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[:, 0, 1:].reshape(-1, grid_size, grid_size)
    return mask


def valid(args, model, writer, test_loader, epoch, is_normal=True):
    # Validation!
    eval_losses = AverageMeter()
    eval_losses_adv = AverageMeter()
    att_losses = AverageMeter()
    att_rollout_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label, all_preds_adv = [], [], []
    all_attn_loss, all_attn_rollout_loss = [], []
    all_max_softmax, all_max_softmax_adv = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X) (att_loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0],
                          position=0, leave=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        noised_x = pgd_attack(x, model, eps=args.pgd_eps, n_iter=args.pgd_iter)
        model.zero_grad()
        model.eval()
        if step == 0:
            visualize(x.clone(), noised_x.clone(), epoch, is_normal)
        with torch.no_grad():
            logits, attn_weights = model(x)
            attn_stack = torch.stack(attn_weights, dim=1)

            logits_adv, noisy_attn = model(noised_x)
            noisy_attn_stack = torch.stack(noisy_attn, dim=1)

            attn_diff = (attn_stack - noisy_attn_stack) ** 2
            att_losses.update(attn_diff.mean().item())
            batch_attn = torch.mean(attn_diff, tuple(range(1, attn_diff.ndim))).detach().cpu().numpy()
            if len(all_attn_loss) == 0:
                all_attn_loss.append(batch_attn)
            else:
                all_attn_loss[0] = np.append(
                    all_attn_loss[0], batch_attn, axis=0
                )

            att_mask = get_att_mask(attn_weights)
            noisy_mask = get_att_mask(noisy_attn)
            if step == 0:
                visualize_attention(x, att_mask, epoch, is_normal, "Pics_Attn_Normal")
                visualize_attention(noised_x, noisy_mask, epoch, is_normal, "Pics_Attn_Attacked")

            attn_rollout_diff = (att_mask - noisy_mask) ** 2
            att_rollout_losses.update(attn_rollout_diff.mean().item())
            batch_attn = torch.mean(attn_rollout_diff, tuple(range(1, attn_rollout_diff.ndim))).detach().cpu().numpy()
            if len(all_attn_rollout_loss) == 0:
                all_attn_rollout_loss.append(batch_attn)
            else:
                all_attn_rollout_loss[0] = np.append(
                    all_attn_rollout_loss[0], batch_attn, axis=0
                )

            if is_normal:
                eval_loss = loss_fct(logits, y)
                eval_losses.update(eval_loss.item())

                eval_loss_adv = loss_fct(logits_adv, y)
                eval_losses_adv.update(eval_loss_adv.item())

            softmax = torch.nn.functional.softmax(logits, dim=1)
            max_softmax = torch.max(softmax, 1).values.detach().cpu().numpy()
            if len(all_max_softmax) == 0:
                all_max_softmax.append(max_softmax)
            else:
                all_max_softmax[0] = np.append(
                    all_max_softmax[0], max_softmax, axis=0
                )

            softmax_adv = torch.nn.functional.softmax(logits_adv, dim=1)
            max_softmax_adv = torch.max(softmax_adv, 1).values.detach().cpu().numpy()
            if len(all_max_softmax_adv) == 0:
                all_max_softmax_adv.append(max_softmax_adv)
            else:
                all_max_softmax_adv[0] = np.append(
                    all_max_softmax_adv[0], max_softmax_adv, axis=0
                )

            preds = torch.argmax(logits, dim=-1)
            adv_preds = torch.argmax(logits_adv, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_preds_adv.append(adv_preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_preds_adv[0] = np.append(
                all_preds_adv[0], adv_preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description(
            "Validating... (loss=%2.5f) (att_loss=%2.6f)" % (eval_losses.avg, att_losses.avg))

    all_preds, all_preds_adv, all_label = all_preds[0], all_preds_adv[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy_adv = simple_accuracy(all_preds_adv, all_label)
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Epoch: %d" % epoch)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Attention Loss: %2.6f" % att_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=epoch)
    if is_normal:
        return accuracy, accuracy_adv, att_losses.avg, att_rollout_losses.avg, all_attn_loss[0], \
               all_attn_rollout_loss[0], all_max_softmax[0], all_max_softmax_adv[
                   0], eval_losses.avg, eval_losses_adv.avg
    else:
        return att_losses.avg, att_rollout_losses.avg, all_attn_loss[0], all_attn_rollout_loss[0], all_max_softmax[0], \
               all_max_softmax_adv[0]


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

        adv_x = adv_x + (2.5 / n_iter) * eps * adv_x.grad.sign()
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
    normal_train_loader, normal_val_loader, normal_test_loader, \
    outlier_train_loader, outlier_val_loader, outlier_test_loader = get_outlier_loader(args)

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
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

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
    if args.training_weights_dir:
        epoch, normal_train_loader = load_model_training(args, model, optimizer, scheduler)
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
            _, attn_weights = model(x, y)
            attn_mask = get_att_mask(attn_weights)
            if step == 0:
                visualize_attention(x, attn_mask, epoch, True, "Pics_Train_Normal")
            attn_weights = torch.stack(attn_weights, dim=1)
            noised_x = pgd_attack(x, model, eps=args.pgd_eps, n_iter=args.pgd_iter)
            model.train()
            loss, noisy_attn = model(noised_x, y)
            attn_mask_adv = get_att_mask(noisy_attn)
            if step == 0:
                visualize_attention(noised_x, attn_mask_adv, epoch, True, "Pics_Train_Adv")
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
                    "Training (%d / %d Steps) (loss=%2.5f) (att_loss=%2.6f)" % (
                        epoch, t_total, losses.avg, attentions.avg)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=epoch)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=epoch)

        optimizer.zero_grad()
        if args.local_rank in [-1, 0]:
            epoch += 1

            accuracy, accuracy_adv, normal_attn_loss, normal_rollout_loss, normal_all_attn_loss, normal_all_attn_rollout_loss, \
            normal_all_max_softmax, normal_all_max_softmax_adv, eval_loss, eval_losses_adv = valid(args, model, writer,
                                                                                            normal_val_loader, epoch)

            outlier_attn_loss, outlier_att_rollout_loss, outlier_all_attn_loss, outlier_all_attn_rollout_loss, \
            outlier_all_max_softmax, outlier_all_max_softmax_adv = valid(args, model, writer, outlier_val_loader, epoch,
                                                                         is_normal=False)

            attn_auc = roc_auc_score([0] * len(normal_all_attn_loss) + [1] * len(outlier_all_attn_loss),
                                     np.append(normal_all_attn_loss, outlier_all_attn_loss, axis=0))

            attn_rollout_auc = roc_auc_score([0] * len(normal_all_attn_rollout_loss) + [1] * len(outlier_all_attn_rollout_loss),
                                     np.append(normal_all_attn_rollout_loss, outlier_all_attn_rollout_loss, axis=0))

            max_softmax_auc = roc_auc_score([0] * len(normal_all_max_softmax) + [1] * len(outlier_all_max_softmax),
                                     np.append(normal_all_max_softmax, outlier_all_max_softmax, axis=0))

            max_softmax_adv_auc = roc_auc_score([0] * len(normal_all_max_softmax_adv) + [1] * len(outlier_all_max_softmax_adv),
                                     np.append(normal_all_max_softmax_adv, outlier_all_max_softmax_adv, axis=0))

            val_csv_writer.add_record([epoch, accuracy, accuracy_adv, normal_attn_loss, normal_rollout_loss, outlier_attn_loss,
                                       outlier_att_rollout_loss, attn_auc, attn_rollout_auc, max_softmax_auc, max_softmax_adv_auc, eval_loss, eval_losses_adv])
            logger.info('AUC Score: %.5f' % max_softmax_auc)
            save_model_training(args, epoch, model, optimizer, scheduler, normal_train_loader)
            if best_acc < accuracy:
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
    parser.add_argument("--pretrained_dir", type=str,  # default="checkpoint/ViT-B_16.npz",
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
    parser.add_argument("--training_weights_dir", type=str,
                        help="Where to search for pretrained ViT models to continue training.")

    args = parser.parse_args()

    global train_csv_writer
    global val_csv_writer

    train_csv_writer = CSV_Writer(path=args.csv_path + args.name + '_train.csv',
                                  columns=(['epoch', 'train_loss', 'train_attention_loss']))

    val_csv_writer = CSV_Writer(path=args.csv_path + args.name + '_val.csv',
                                columns=(['epoch', 'accuracy', 'accuracy_adv', 'normal_attn_loss', 'normal_rollout_loss'
                                          'outlier_attn_loss', 'outlier_att_rollout_loss', 'attn_auc', 'attn_rollout_auc',
                                          'max_softmax_auc', 'max_softmax_adv_auc', 'eval_loss', 'eval_losses_adv']))

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
