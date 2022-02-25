# modification based on https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder

from PIL import Image

import os
import os.path

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

import copy
import warnings

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageFolder(Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        label_file (string): a csv file containing the class taxonomy
        label_space (list): a list of strings used to specify the columns in label file 
                    to be used as the labels. The order of the list specifies the coarsity of 
                    the labels. The last string in the label_space should specify the finest labels
        loader (callable): A function to load a sample given its path. 
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        base_taxonomy: the class taxonomy/hierarchy of another dataset. If specified, 
        then whole taxonomy (except the leaf nodes) will be used as the hierarchy. 
        Leaf nodes that does not have a parent in the base_taxonomy will be assigned a label of -1

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, label_file, label_space, 
            loader=default_loader, 
            transform=None, target_transform=None,
            base_taxonomy=None):
        super(ImageFolder, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

        self.label_file = pd.read_csv(label_file)
        self.label_space = label_space

        self.loader = loader
        
        # List: list[i] - labels for the i-th level
        if base_taxonomy is None:
            self.class_hierarchy = [list(np.sort(np.unique(self.label_file[i]))) for i in label_space]
        else:
            self.class_hierarchy = copy.deepcopy(base_taxonomy)
            self.class_hierarchy[-1] = list(
                np.sort(np.unique(self.label_file[self.label_space[-1]])))

        # List: list[i] - index to label for the i-th level
        self.class_to_idx_hierarchy = [ {i: ind for ind, i in enumerate(cl)} for cl in self.class_hierarchy]

        # List: list[i] - children of the parent at the i-th level
        # list[i][j] returns the children of class j (where j is at the i-th level)
        self.lvl_child = []
        for ind, l in enumerate(label_space[:-1]):
            children = {}
            for j in self.class_hierarchy[0]:
                children[j] = sorted(list(self.label_file.loc[self.label_file[l] == j][label_space[ind + 1]]))
            
            self.lvl_child.append(children)

        self._make_dataset()
        if torch.isclose(self.targets[:, -1].float().mean(), torch.ones(1) * -1):
            warnings.warn("All the immediate parent are not found!")

    def _make_dataset(self):
        """
        Construct of dataset based on 
        """

        samples = []
        targets = []

        for _, row in self.label_file.iterrows():
            
            temp = [os.path.join(str(row[self.label_space[-1]]), j) for j in os.listdir(
                            os.path.join(self.root, str(row[self.label_space[-1]])))]
            samples += temp

            label = []

            for ind, i in enumerate(self.label_space):
                if row[i] not in self.class_hierarchy[ind]:
                    label.append(-1) # assign a label of -1 if no ancestors are found
                else:
                    label.append(self.class_to_idx_hierarchy[ind][row[i]])
                
            targets += ([label] * len(temp))

        self.samples = samples
        self.targets = torch.LongTensor(targets)
        
        # the following variable is created for compatibility issue
        self.imgs = [[self.samples[i], self.targets[i]]
                     for i in range(len(self.samples))]

        return 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        target = self.targets[index]
        
        sample = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)
