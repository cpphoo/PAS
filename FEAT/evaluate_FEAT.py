from utils import accuracy
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np
from tqdm import tqdm

import models

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

from PIL import Image
import datetime

import random

from feat_models.models import FEAT


class indexed_dataset(Dataset):
    '''
    This class is a dataset class that takes in a categories file and construct a dataset
    based on the category specified in the category file 
    This is useful when we only want to evaluate certain subset of classes 
    '''

    def __init__(self, root, categories, label_space, parent_idx, transforms=None):
        '''
            label_space is a list of two items where the first is the group name of the parent
            e.g ['supercategory', 'name']
            parent idx a mapping from supercategory to index
        '''
        super(indexed_dataset, self).__init__()
        self.transforms = transforms
        self.categories = categories
        self.root = root

        self.samples = []
        self.labels = []
        self.lvl_children = {i: [] for i in parent_idx}

        # construct samples
        for i in range(len(categories)):
            info = categories.iloc[i]
            dir_path = os.path.join(root, str(info['id']))

            # get the parent class
            parent = info[label_space[-2]]
            parent_label = parent_idx[parent]
            self.lvl_children[parent].append(i)

            imgs = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
            self.samples += imgs
            self.labels += [[parent_label, i]] * len(imgs)
        self.labels = torch.LongTensor(self.labels)

    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        path = self.samples[index]
        img = self._pil_loader(path)

        if transforms is None:
            return img, self.labels[index]
        else:
            return self.transforms(img), self.labels[index]

    def __len__(self):
        return len(self.samples)


def load_model(model_dict, ckpt):
    sd = torch.load(ckpt)

    for i in model_dict:
        model_dict[i].load_state_dict(sd[i])
    return model_dict


def generate_prototypes(reps, labels, lvl_children, num_max_parents_per_task, num_shots, num_ways, cosine=False):
    '''
        Generate prototypes for a single episode

        reps: nxd matrix, representation of the images
        labels: n vector, the label of each vector in reps
        num_max_parents_per_task: the number of allowable parents in a single episode
        num_shots: number of reference examples
        num_ways: number of classes for the episode
        cosine: whether to normalize the prototypes
    '''
    prototypes = []
    prototypes_label = []

    # sample parents
    parent_idx = list(lvl_children.keys())
    np.random.shuffle(parent_idx)
    parent_idx = parent_idx[:num_max_parents_per_task]

    children_classes = torch.cat([
        torch.LongTensor(lvl_children[i]) for i in parent_idx
    ], dim=0)

    categories = children_classes[torch.randperm(
        n=len(children_classes))[:num_ways]]

    for i in categories:
        ind = torch.nonzero(labels[:, -1] == i, as_tuple=False).squeeze()

        if num_shots > 0:
            sample = torch.randperm(len(ind))[:num_shots]
        else:
            sample = torch.arange(len(ind))

        ind_l = ind[sample.cuda()]

        prototypes.append(torch.mean(reps[ind_l], dim=0))

        prototypes_label.append(labels[ind_l[0]])

    prototypes = torch.stack(prototypes)

    # need to normalize the prototypes again after addition
    if cosine:
        prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-12)

    prototypes_label = torch.stack(prototypes_label, dim=0)

    return prototypes, prototypes_label


def generate_logits(query, centroids, cosine=False):
    '''
        Helper function to generate logits
    '''
    if cosine:
        metric = 'cosine_with_normalized_features'
    else:
        metric = 'squared_l2'

    logits = models.predict(centroids=centroids, query=query, metric=metric)
    return logits


def generate_reps(model, dloader, cosine=False):
    '''
        Do a forward pass on model on all the data in dloader
    '''
    reps = []
    labels = []

    for X, y in tqdm(dloader):
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.no_grad():
            rep = model(X)
        reps.append(rep)
        labels.append(y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)

    if cosine:
        reps = F.normalize(reps, dim=1, p=2, eps=1e-12)
    return reps, labels


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Need to create dataloader
    # load indices for novel categories
    # base_categories = pd.read_csv(args.base_categories)
    novel_categories = pd.read_csv(args.novel_categories)
    parent_names = list(
        np.sort(np.unique(novel_categories[args.label_space[-2]])))
    num_parent = len(parent_names)

    parent_idx = {i: ind for ind, i in enumerate(parent_names)}

    num_novel = len(novel_categories)

    if args.image_transformations == 'standard':
        transforms_final = transforms.Compose([
            transforms.CenterCrop(84),
            transforms.ToTensor(),
        ])
    elif args.image_transformations == 'standard_cifar':
        transforms_final = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('Invalid Image Transformation!')

    ref_novel_dset = indexed_dataset(
        args.ref_path, novel_categories, args.label_space, parent_idx,
        transforms_final)

    query_novel_dset = indexed_dataset(
        args.query_path, novel_categories, args.label_space, parent_idx,
        transforms_final)

    ref_novel_dloader = DataLoader(ref_novel_dset, batch_size=args.bsize,
                                   num_workers=args.num_workers,
                                   shuffle=False, drop_last=False, pin_memory=True)

    query_novel_dloader = DataLoader(query_novel_dset, batch_size=args.bsize,
                                     num_workers=args.num_workers,
                                     shuffle=False, drop_last=False, pin_memory=True)

    # Create model
    sd = torch.load(args.ckpt)

    model_dict = {}
    backbone = models.resnet18(
        remove_last_relu=args.remove_last_relu, high_res_input=False)
    feature_dim = 512


    backbone = nn.DataParallel(backbone.cuda())
    feat_model = FEAT(sd['args'], backbone, feature_dim)

    # load the model
    model_dict['model'] = feat_model.cuda()

    # detach the model
    for i in model_dict:
        for p in model_dict[i].parameters():
            p.requires_grad_(False)

    model_dict = load_model(model_dict, args.ckpt)
    model = model_dict['model'].encoder.eval()
    self_attn = model_dict['model'].slf_attn.eval()

    result_average = [
        {str(s): np.zeros(args.num_repetitions) for s in args.shots}
    ]

    result_per_class = [
        {str(s): np.zeros(args.num_repetitions) for s in args.shots}
    ]

    # name for each tab
    # naming: all and parent_classes
    names = ['novel_all']  # + ['novel_' + p for p in parent_names]

    # generate embeddings for the query examples
    reps_ref_novel, labels_ref_novel = generate_reps(
        model, ref_novel_dloader, args.cosine)

    # This might take up a lot of the gpu memory
    # Move this to cpu for now and then when prototypes are generated,
    # move those prototypes to gpu
    reps_ref_novel = reps_ref_novel.cpu()

    # generate embeddings for the query examples
    reps_query_novel, labels_query_novel = generate_reps(
        model, query_novel_dloader, args.cosine)

    reps_query_novel = reps_query_novel.cuda()

    for s in args.shots:
        num_repetitions = args.num_repetitions

        for i in tqdm(range(num_repetitions)):
            # generate prototypes
            prototypes_novel, prototypes_label = generate_prototypes(
                reps_ref_novel, labels_ref_novel, ref_novel_dset.lvl_children,
                args.num_max_parents_per_task, s, args.ways,
                args.cosine)

            prototypes_novel = prototypes_novel.unsqueeze(0).cuda()
            prototypes_novel = self_attn(
                prototypes_novel, prototypes_novel, prototypes_novel)
            prototypes_novel = prototypes_novel[0]

            query_ind = torch.nonzero(labels_query_novel[:, -1].view(-1, 1) == prototypes_label[:, -1].cuda(),
                                      as_tuple=False)

            # relabel the query label to [0, ways)
            labels_query_novel_relabel = query_ind[:, -1]

            # Evaluate on query
            logits_novel = generate_logits(
                reps_query_novel[query_ind[:, 0]], prototypes_novel, args.cosine)
            # labels for all class prediction

            # for novel_all
            t = accuracy(logits_novel, labels_query_novel_relabel)

            result_average[0][str(s)][i] = t['average'][0].item()
            result_per_class[0][str(s)][i] = t['per_class_average'][0].item()

        if args.result_file is None:
            result_file = "result_" + str(datetime.datetime.now())
        else:
            result_file = args.result_file

        # print(result_file)
        with pd.ExcelWriter(result_file + '.xlsx', engine='xlsxwriter') as f:
            for ind, i in enumerate(names):
                pd.DataFrame(result_per_class[ind]).to_excel(
                    f, sheet_name=('per_class_' + i)[:31])

            for ind, i in enumerate(names):
                pd.DataFrame(result_average[ind]).to_excel(
                    f, sheet_name=('average_' + i)[:31])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--ref_path', type=str, required=True,
                        help='path to all reference images')
    parser.add_argument('--query_path', type=str,
                        required=True, help='path to all query images')
    parser.add_argument('--image_transformations', type=str,
                        default=None, help='Image Transformations to use')
    parser.add_argument('--novel_categories', type=str, required=True,
                        help='path to a csv file that has the novel categories')
    parser.add_argument('--label_space', type=str, nargs='+', default=['supercategory', 'name'],
                        help='label space to consider')
    parser.add_argument('--bsize', type=int, default=512,
                        help='batch_size for loading data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of threads for dataloading')

    # models
    parser.add_argument('--ckpt', type=str, required=True,
                        help='model to evaluate')
    parser.add_argument('--cosine', action='store_true',
                        help='use cosine classifier')
    parser.add_argument('--remove_last_relu', action='store_true',
                        help='whether to remove the last relu activation for resnet')

    # Bookeeping
    parser.add_argument('--result_file', type=str,
                        help='Filename for the result')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')

    # Task specification
    parser.add_argument('--num_repetitions', type=int, default=10,
                        help='num of repeats')
    parser.add_argument('--shots', type=int, nargs='+', default=[0, 1, 2, 5, 10],
                        help='the number of shots')
    parser.add_argument('--ways', type=int, default=5,
                        help='number of ways')
    parser.add_argument('--num_max_parents_per_task', type=int, default=3,
                        help='number of ways')

    args = parser.parse_args()
    main(args)
