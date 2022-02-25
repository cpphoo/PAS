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

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.bsize, 
                                              shuffle=True, 
                                              drop_last=True,
                                              num_workers=args.num_workers, 
                                              pin_memory=True)
    ############################

    ###########################
    # Create Models
    ###########################
    backbone = models.resnet18(
        remove_last_relu=args.remove_last_relu, high_res_input=False)
    feature_dim = 512


    backbone = nn.DataParallel(backbone).cuda()
    num_base_classes = len(trainset.class_hierarchy[-1])

    logger.info(f"num_classes: {num_base_classes}")

    if args.use_cosine_clf:
        if not args.remove_last_relu:
            warnings.warn(
                "Using cosine classifier without the last relu activation removed!")
        clf = models.cosine_clf(feature_dim, num_base_classes).cuda()
    else:
        if args.remove_last_relu:
            warnings.warn(
                "Using linear classifier with the last relu activation removed!")
        clf = nn.Linear(feature_dim, num_base_classes).cuda()
    ############################

    ###########################
    # Create Optimizer
    ###########################

    optimizer = torch.optim.SGD([
        {'params': backbone.parameters()},
        {'params': clf.parameters()}
    ],
        lr=args.lr, momentum=0.9,
        weight_decay=args.wd,
        nesterov=False)
    ############################

    ############################
    # Specify Loss Function
    ############################
    criterion = nn.NLLLoss(reduction='mean')
    ############################
    starting_epoch = 0

    if args.load_path is not None:
        print('Loading model from {}'.format(args.load_path))
        logger.info('Loading model from {}'.format(args.load_path))
        starting_epoch = load_checkpoint(
            backbone, clf, optimizer, args.load_path)

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
                    backbone, clf, optimizer, load_path)

    # save the initialization
    checkpoint(backbone, clf, optimizer, trainset.class_hierarchy, 
                trainset.class_to_idx_hierarchy, 
    os.path.join(
        args.dir, f'checkpoint_{starting_epoch}.pkl'), starting_epoch)

    try:
        for epoch in tqdm(range(starting_epoch, args.epochs)):
            train(backbone, clf, optimizer, trainloader, criterion,
                         epoch, args.epochs, logger, trainlog, args)

            # Always checkpoint after first epoch of training
            if (epoch == starting_epoch) or ((epoch + 1) % args.save_freq == 0):
                checkpoint(backbone, clf, optimizer,  trainset.class_hierarchy, 
                trainset.class_to_idx_hierarchy,
                os.path.join(
                    args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)

        if (epoch + 1) % args.save_freq != 0:
            checkpoint(backbone, clf, optimizer,  trainset.class_hierarchy,
                       trainset.class_to_idx_hierarchy, os.path.join(
                args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)
    finally:
        trainlog.save()
    return


def checkpoint(backbone, clf, optimizer, class_hierarchy, class_to_idx_hierarchy, 
            save_path, epoch):
    '''
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    '''
    torch.save({
        'backbone': backbone.module.state_dict(),
        'clf': clf.state_dict(),
        'opt': optimizer.state_dict(),
        'class_hierarchy': class_hierarchy,
        'class_to_idx_hierarchy': class_to_idx_hierarchy, 
        'epoch': epoch
    }, save_path)
    return


def load_checkpoint(backbone, clf, optimizer, load_path):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path)
    backbone.module.load_state_dict(sd['backbone'])
    clf.load_state_dict(sd['clf'])
    optimizer.load_state_dict(sd['opt'])
    return sd['epoch']


def lr_schedule_step(optimizer, epoch, step_in_epoch, total_steps_in_epochs, args):
    '''
        Warm up the learning rate by 1 epoch

        Drops the learning rate by a factor of 10 every one-third of the total number
        of training epochs
    '''
    step = epoch + step_in_epoch / total_steps_in_epochs

    def linear_rampup(current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length

    #### linear rampup for the first epochs
    lr = linear_rampup(step, 1) * args.lr

    #### dropdown the learning rate by a factor of 10 every lr decay freq epochs
    def step_rampdown(epoch):
        return 0.1 ** (epoch // args.lr_decay_freq)

    lr = step_rampdown(epoch) * lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train(backbone, clf, optimizer, trainloader, criterion, 
                epoch, num_epochs, logger, trainlog, args):
    '''
        Train the classifier (backbone, clf) by one epoch
    '''
    meters = utils.AverageMeterSet()
    backbone.train()
    clf.train()

    start_data = torch.cuda.Event(enable_timing=True)
    end_data = torch.cuda.Event(enable_timing=True)

    start_update = torch.cuda.Event(enable_timing=True)
    end_update = torch.cuda.Event(enable_timing=True)

    end = time.perf_counter()
    torch.cuda.synchronize()
    start_data.record()
    for i, (X, y) in enumerate(trainloader):
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        end_data.record()

        start_update.record()
        lr = lr_schedule_step(optimizer, epoch, i, len(trainloader), args)
        
        optimizer.zero_grad()
        features = backbone(X)
        logits = clf(features)

        perf = utils.accuracy(logits.data, y[:, -1].data, topk=(1,5), 
                    compute_mean_class=False)
        log_probability = F.log_softmax(logits, dim=1)
        loss = criterion(log_probability, y[:, -1])
        loss.backward()
        optimizer.step()

        end_update.record()

        torch.cuda.synchronize()
        meters.update('Iteration_time', time.perf_counter() - end, 1)
        meters.update('Data_time', start_data.elapsed_time(end_data) * 1e-3, 1)
        meters.update('Update_time', start_update.elapsed_time(end_update) * 1e-3, 1)
        meters.update('Loss', loss.item(), 1)
        meters.update('Lr', lr, 1)
        meters.update('top1', perf['average'][0].item(), 1)
        meters.update('top5', perf['average'][1].item(), 1)

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
               'top5': averages['top5/avg'],
              }, step=epoch+1)

    return averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to Training a Image Classification Model')
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

    # model
    parser.add_argument('--remove_last_relu', action='store_true',
                        help='whether to remove the last relu activation for resnet')
    parser.add_argument('--use_cosine_clf', action='store_true',
                        help='whether to use cosine classifier')


    # optimization
    parser.add_argument('--bsize', type=int, default=256,
                        help='batch_size for labeled data')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate for model')
    parser.add_argument('--lr_decay_freq', type=int, default=100,
                        help='learning rate decay frequency (in epochs)')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='Weight decay for the model')


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
