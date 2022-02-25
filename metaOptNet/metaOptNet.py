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

from metaOptNet_models.classification_heads import ClassificationHead


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

    embedding_net = models.dataparallel_wrapper(backbone)
    embedding_net = nn.DataParallel(embedding_net).cuda()

    clf_head = ClassificationHead(base_learner='SVM-CS').cuda()

    num_base_classes = len(trainset.class_hierarchy[-1])

    logger.info(f"num_classes: {num_base_classes}")

    
    ############################

    ###########################
    # Create Optimizer
    ###########################
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params': clf_head.parameters()}], lr=args.lr,
                                momentum=args.mom,
                                weight_decay=args.weight_decay, nesterov=True)

    def lambda_epoch(e): return 1.0 if e < 20 else (
        0.06 if e < 40 else 0.012 if e < 50 else (0.0024))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    ############################

    ############################
    # Specify Loss Function
    ############################
    criterion = nn.CrossEntropyLoss(reduction='mean')
    ############################
    starting_epoch = 0

    if args.load_path is not None:
        print('Loading model from {}'.format(args.load_path))
        logger.info('Loading model from {}'.format(args.load_path))
        starting_epoch = load_checkpoint(
            embedding_net, clf_head, optimizer, lr_scheduler, args.load_path)

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
                    embedding_net, clf_head, optimizer, lr_scheduler, load_path)

    # save the initialization
    checkpoint(embedding_net, clf_head, optimizer, lr_scheduler, trainset.class_hierarchy,
                trainset.class_to_idx_hierarchy, 
    os.path.join(
        args.dir, f'checkpoint_{starting_epoch}.pkl'), starting_epoch)

    try:
        for epoch in tqdm(range(starting_epoch, args.epochs)):

            # generate new batches of episodes
            trainloader = episodic_trainset.generate_loader(
                batch_size=args.bsize,
                shuffle=False, drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True)


            train(embedding_net, clf_head, optimizer, trainloader, criterion,
                         epoch, args.epochs, logger, trainlog, args)
            
            lr_scheduler.step()

            # Always checkpoint after first epoch of training
            if (epoch == starting_epoch) or ((epoch + 1) % args.save_freq == 0):
                checkpoint(embedding_net, clf_head, optimizer, lr_scheduler,  trainset.class_hierarchy,
                trainset.class_to_idx_hierarchy,
                os.path.join(
                    args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)

        if (epoch + 1) % args.save_freq != 0:
            checkpoint(embedding_net, clf_head, optimizer, lr_scheduler,  trainset.class_hierarchy,
                       trainset.class_to_idx_hierarchy, os.path.join(
                args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)
    finally:
        trainlog.save()
    return


def checkpoint(backbone, clf, optimizer, scheduler, 
                class_hierarchy, class_to_idx_hierarchy, 
                save_path, epoch):
    '''
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    '''
    torch.save({
        'backbone': backbone.module.state_dict(),
        'clf': clf.state_dict(),
        'opt': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 
        'class_hierarchy': class_hierarchy,
        'class_to_idx_hierarchy': class_to_idx_hierarchy, 
        'epoch': epoch
    }, save_path)
    return


def load_checkpoint(backbone, clf, optimizer, scheduler, load_path):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path)
    backbone.module.load_state_dict(sd['backbone'])
    clf.load_state_dict(sd['clf'])
    optimizer.load_state_dict(sd['opt'])
    scheduler.load_state_dict(sd['scheduler'])
    return sd['epoch']

# ported from https://github.com/kjunelee/MetaOptNet/blob/master/train.py
def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies

def train(embedding_net, clf_head, optimizer, trainloader, criterion, 
                epoch, num_epochs, logger, trainlog, args):
    '''
        Train the classifier (backbone, clf) by one epoch
    '''
    meters = utils.AverageMeterSet()
    embedding_net.train()
    clf_head.train()

    num_support = args.way * args.shot
    num_query = args.way * args.query

    start_data = torch.cuda.Event(enable_timing=True)
    end_data = torch.cuda.Event(enable_timing=True)

    start_update = torch.cuda.Event(enable_timing=True)
    end_update = torch.cuda.Event(enable_timing=True)

    end = time.perf_counter()
    torch.cuda.synchronize()
    start_data.record()
    for i, (Xref, yref, Xquery, yquery) in enumerate(trainloader):
        Xref = Xref.cuda(non_blocking=True)
        yref = yref.cuda(non_blocking=True)

        Xquery = Xquery.cuda(non_blocking=True)
        yquery = yquery.cuda(non_blocking=True)

        end_data.record()

        start_update.record()
        lr = optimizer.param_groups[0]['lr']

        num_episodes = Xref.shape[0]
        image_dimension = list(Xref.shape)[2:]

        # Collapse all the episodes into one
        X_ref_flatten = Xref.view(-1, *image_dimension)
        X_query_flatten = Xquery.view(-1, *image_dimension)
        
        optimizer.zero_grad()
        # MetaOpt Net operates on the features maps, rather than global averaged pooled features
        features_ref = embedding_net('feature_maps', X_ref_flatten).view(
            X_ref_flatten.size(0), -1)
        features_ref = features_ref.view(num_episodes, num_support, -1)

        features_query = embedding_net('feature_maps', X_query_flatten).view(
            X_query_flatten.size(0), -1)
        features_query = features_query.view(num_episodes, num_query, -1)
        
        # relabel the label to [0, Nway]
        yref_relabel = []
        yquery_relabel = []
        for j in range(num_episodes):
            unique_label = yref[j, :, -1].unique()
            yref_relabel.append(torch.nonzero(
                yref[j, :, -1].view(-1, 1) == unique_label, as_tuple=False)[:, -1])
            yquery_relabel.append(torch.nonzero(
                yquery[j, :, -1].view(-1, 1) == unique_label, as_tuple=False)[:, -1])
        
        yref_relabel = torch.stack(yref_relabel)
        yquery_relabel = torch.stack(yquery_relabel)

        logits_query = clf_head(
            features_query, features_ref, yref_relabel, args.way, args.shot)
        logits_query = logits_query.view(-1, args.way)

        smoothed_one_hot = one_hot(yquery_relabel.view(-1), args.way)
        smoothed_one_hot = smoothed_one_hot * \
            (1 - args.eps) + (1 - smoothed_one_hot) * \
            args.eps / (args.way - 1)

        log_prb = F.log_softmax(
            logits_query, dim=1)
        loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        loss = loss.mean()

        perf = utils.accuracy(
            logits_query, yquery_relabel.view(-1).data, topk=(1,), 
            compute_mean_class=False)

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
        description='Script to Training MetaOptNet')
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
    parser.add_argument('--use_cosine_clf', action='store_true',
                        help='whether to use cosine classifier')


    # optimization
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--bsize', type=int, default=1,
                        help='number of episode per batch')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--eps', type=float, default=0.1,
                        help='smoothing parameters for the ground truth label')
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
