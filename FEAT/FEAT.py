import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from tqdm import tqdm
import argparse

import os
import numpy as np
import random

import utils

import models
import data
import time

import wandb
import warnings

from feat_models.models import FEAT

import torch.optim as optim


def main(args):
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    logger = utils.create_logger(os.path.join(
        args.dir, 'checkpoint.log'), __name__)
    trainlog = utils.savelog(args.dir, 'train')

    wandb.init(project='few_shot_coarse_labels',
               group=__file__,
               name=f'{__file__}_{args.dir}')

    wandb.config.update(args)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ###########################
    # Create DataLoader
    ###########################
    if args.image_transformations == 'standard':
        transforms_final = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif args.image_transformations == 'standard_cifar':
        transforms_final = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        raise ValueError('Invalid Image Transformation!')

    trainset = data.ImageFolder(root=args.trainpath,
                                label_file=args.label_file,
                                label_space=args.label_space,
                                transform=transforms_final)
    num_base_classes = len(trainset.class_hierarchy[-1])

    episodic_trainset = data.Episodic_wrapper(trainset,
                                              num_way=args.way,
                                              num_ref=args.shot,
                                              num_query=args.query,
                                              num_episodes=args.episodes_per_epoch)
    ############################

    ###########################
    # Create Models
    ###########################
    backbone = models.resnet18(
        remove_last_relu=args.remove_last_relu, high_res_input=False)
    feature_dim = 512

    if args.pretrained_embedding_weights is None:
        warnings.warn("Training Feat with Untrained Network")
    else:
        sd = torch.load(args.pretrained_embedding_weights)
        backbone.load_state_dict(sd['backbone'])

    backbone = nn.DataParallel(backbone.cuda())

    feat_model = FEAT(args, backbone, feature_dim)
    feat_model = feat_model.cuda()

    logger.info(f"num_classes: {num_base_classes}")

    ############################

    ###########################
    # Create Optimizer
    ###########################
    optimizer, scheduler = prepare_optimizer(feat_model, args)
    ############################

    starting_epoch = 0

    if args.load_path is not None:
        print('Loading model from {}'.format(args.load_path))
        logger.info('Loading model from {}'.format(args.load_path))
        starting_epoch = load_checkpoint(
            feat_model, optimizer, scheduler, args.load_path)

    if args.resume_latest:
        # Only works if model is saved as checkpoint_(/d+).pkl
        import re
        pattern = "checkpoint_(\d+).pkl"
        candidate = []
        for i in os.listdir(args.dir):
            match = re.search(pattern, i)
            if match:
                candidate.append(int(match.group(1)))

        # if nothing found, then start from scratch
        if len(candidate) == 0:
            print('No latest candidate found to resume!')
            logger.info('No latest candidate found to resume!')
        else:
            latest = np.amax(candidate)
            load_path = os.path.join(args.dir, f'checkpoint_{latest}.pkl')
            if latest >= args.epochs:
                print('The latest checkpoint found ({}) is after the number of epochs (={}) specified! Exiting!'.format(
                    load_path, args.epochs))
                logger.info('The latest checkpoint found ({}) is after the number of epochs (={}) specified! Exiting!'.format(
                    load_path, args.epochs))
                import sys
                sys.exit(0)
            else:
                print('Resuming from the latest checkpoint: {}'.format(load_path))
                logger.info(
                    'Resuming from the latest checkpoint: {}'.format(load_path))
                if args.load_path:
                    logger.info(
                        'Overwriting model loaded from {}'.format(args.load_path))
                starting_epoch = load_checkpoint(
                    feat_model, optimizer, scheduler, load_path)

    # save the initialization
    checkpoint(feat_model, optimizer, scheduler, trainset.class_hierarchy,
               trainset.class_to_idx_hierarchy,
               os.path.join(
                   args.dir, f'checkpoint_{starting_epoch}.pkl'), starting_epoch, args)

    try:
        for epoch in tqdm(range(starting_epoch, args.epochs)):

            # generate new batches of episodes
            trainloader = episodic_trainset.generate_loader(
                batch_size=args.bsize,
                shuffle=False, drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True)

            train(feat_model, optimizer, trainloader, 
                  epoch, args.epochs, logger, trainlog, args)

            scheduler.step()

            # Always checkpoint after first epoch of training
            if (epoch == starting_epoch) or ((epoch + 1) % args.save_freq == 0):
                checkpoint(feat_model, optimizer, scheduler,  trainset.class_hierarchy,
                           trainset.class_to_idx_hierarchy,
                           os.path.join(
                               args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1, args)

        if (epoch + 1) % args.save_freq != 0:
            checkpoint(feat_model, optimizer, scheduler,  trainset.class_hierarchy,
                       trainset.class_to_idx_hierarchy, os.path.join(
                           args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1, args)
    finally:
        trainlog.save()
    return

# ported from https://github.com/Sha-Lab/FEAT/blob/47bdc7c1672e00b027c67469d0291e7502918950/model/trainer/helpers.py
def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    # if args.backbone_class == 'ConvNet':
    #     optimizer = optim.Adam(
    #         [{'params': model.encoder.parameters()},
    #          {'params': top_para, 'lr': args.lr * args.lr_mul}],
    #         lr=args.lr,
    #         # weight_decay=args.weight_decay, do not use weight_decay here
    #     )
    # else:
    optimizer = optim.SGD(
        [{'params': model.encoder.parameters()},
            {'params': top_para, 'lr': args.lr * args.lr_mul}],
        lr=args.lr,
        momentum=args.mom,
        nesterov=True,
        weight_decay=args.weight_decay
    )

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.max_epoch,
            eta_min=0   # a tuning parameter
        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler

def checkpoint(model, optimizer, scheduler,
               class_hierarchy, class_to_idx_hierarchy,
               save_path, epoch, args):
    '''
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    '''
    torch.save({
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'class_hierarchy': class_hierarchy,
        'class_to_idx_hierarchy': class_to_idx_hierarchy,
        'args': args, 
        'epoch': epoch
    }, save_path)
    return


def load_checkpoint(model, optimizer, scheduler, load_path):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path)
    model.load_state_dict(sd['model'])
    optimizer.load_state_dict(sd['opt'])
    scheduler.load_state_dict(sd['scheduler'])
    return sd['epoch']

# ported from https://github.com/kjunelee/MetaOptNet/blob/master/train.py

def prepare_label(args):
    # args = self.args

    # prepare one-hot label
    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
    label_aux = torch.arange(args.way, dtype=torch.int8).repeat(
        args.shot + args.query)

    label = label.type(torch.LongTensor)
    label_aux = label_aux.type(torch.LongTensor)

    if torch.cuda.is_available():
        label = label.cuda()
        label_aux = label_aux.cuda()

    return label, label_aux

def train(feat_model, optimizer, trainloader,
          epoch, num_epochs, logger, trainlog, args):
    '''
        Train the classifier (backbone, clf) by one epoch
    '''
    meters = utils.AverageMeterSet()
    feat_model.train()

    # Prepare one hot label for each episode
    label, label_aux = prepare_label(args)

    start_data = torch.cuda.Event(enable_timing=True)
    end_data = torch.cuda.Event(enable_timing=True)

    start_update = torch.cuda.Event(enable_timing=True)
    end_update = torch.cuda.Event(enable_timing=True)

    end = time.perf_counter()
    torch.cuda.synchronize()
    start_data.record()
    for i, (Xref, yref, Xquery, yquery) in enumerate(trainloader):
        # Wrap the episode into the layout required by FEAT
        # X = (Xref, Xquery)
        # y = (0, 1, ..., # way, 0, 1, .... #way ,...)
        yref_leaf = yref[0, :, -1]
        yquery_leaf = yquery[0, :, -1]

        yref_leaf_argsort = yref_leaf.argsort(descending=False)
        yquery_leaf_argsort = yquery_leaf.argsort(descending=False)

        yref_leaf_argsort = yref_leaf_argsort.view(
            args.way, args.shot).t().reshape(-1)
        yquery_leaf_argsort = yquery_leaf_argsort.view(
            args.way, args.query).t().reshape(-1)

        X = torch.cat([Xref[:, yref_leaf_argsort],
                       Xquery[:, yquery_leaf_argsort]], dim=1)
        y = torch.cat([yref[:, yref_leaf_argsort],
                       yquery[:, yquery_leaf_argsort]], dim=1)

        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)


        end_data.record()

        start_update.record()

        lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        logits, reg_logits = feat_model(X)

        perf = utils.accuracy(logits.data, label.data, topk=(1,))

        if reg_logits is not None:
            loss = F.cross_entropy(logits, label)
            loss_contrastive = F.cross_entropy(reg_logits, label_aux)
            total_loss = loss + args.balance * loss_contrastive

        else:
            loss = F.cross_entropy(logits, label)
            loss_contrastive = torch.zeros(1).cuda()
            total_loss = loss + args.balance * loss_contrastive

        total_loss.backward()
        optimizer.step()

        end_update.record()

        torch.cuda.synchronize()
        meters.update('Iteration_time', time.perf_counter() - end, 1)
        meters.update('Data_time', start_data.elapsed_time(end_data) * 1e-3, 1)
        meters.update('Update_time', start_update.elapsed_time(
            end_update) * 1e-3, 1)


        meters.update('Total_loss', total_loss.item(), 1)
        meters.update('Loss', loss.item(), 1)
        meters.update('Loss_contrastive', loss_contrastive.item(), 1)        
        meters.update('Lr', lr, 1)
        meters.update('top1', perf['average'][0].item(), 1)

        if (i + 1) % args.print_freq == 0:
            logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step} / {steps}] '
                             '{meters:.5f} ').format(
                epoch=epoch, epochs=num_epochs, step=i+1,
                steps=len(trainloader), meters=meters)

            logger.info(logger_string)

        if (args.iteration_bp is not None) and (i+1) == args.iteration_bp:
            break

        end = time.perf_counter()
        start_data.record()

    logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step}] '
                     '{meters:.5f} ').format(
        epoch=epoch+1, epochs=num_epochs, step=0,
        meters=meters)

    logger.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    trainlog.record(epoch, {
        **values,
        **averages,
        **sums
    })

    wandb.log({'loss': averages['Loss/avg']}, step=epoch+1)
    wandb.log({'top1': averages['top1/avg'],
               }, step=epoch+1)

    return averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to Training FEAT')
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to save the checkpoints')

    # dataset and dataloaders
    parser.add_argument('--trainpath', type=str,
                        default='~/data', help='directory to look for the training data')
    parser.add_argument('--label_space', type=str, nargs='+',
                        default=['supercategory', 'name'],
                        help='label space to consider')
    parser.add_argument('--label_file', type=str,
                        help='a csv file containing all the label to be considered')
    parser.add_argument('--image_transformations', type=str,
                        default=None, help='Image Transformations to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)

    # model
    parser.add_argument('--remove_last_relu', action='store_true',
                        help='whether to remove the last relu activation for resnet')
    parser.add_argument('--pretrained_embedding_weights', type=str, default=None)
    parser.add_argument('--use_euclidean', action='store_true', default=False)

    # optimization
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--bsize', type=int, default=1,
                        help='number of episode per batch')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', type=str,
                        default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--balance', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--temperature2', type=float, default=1)
    parser.add_argument('--mom', type=float, default=0.9)

    # miscellany
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Frequency (in epoch) to save')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Frequency (in step per epoch) to print training stats')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume from the latest model in args.dir')
    parser.add_argument('--iteration_bp', type=int,
                        help='number of iteration to break the training loop. Useful for debugging')
    args = parser.parse_args()
    main(args)
