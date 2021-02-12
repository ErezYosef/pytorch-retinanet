#from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader

import skimage.io
#import skimage.transform
import skimage.color
#import skimage



class Infer_Dataset(Dataset):
    """folder dataset."""

    def __init__(self, folder_path, class_list=None, class_n=None, filter_ext=None, transform=None):
        """
        Args:
            folder_path (string): path to folder with images to visualize or inference
            class_list (string, optional): CSV file with class list
            class_n (string, optional): integer of number of classes (insted of class_list)
            filter_ext (string, optional): all file extentions to exclude from the folder files
        """
        self.folder_path = folder_path
        self.image_names = sorted(os.listdir(folder_path))
        #self.train_file = train_file
        self.class_list = class_list
        self.class_n = class_n
        self.transform = transform


        # parse the provided class file
        if self.class_list is not None:
            try:
                with self._open_for_csv(self.class_list) as file:
                    self.classes = self.load_classes(csv.reader(file, delimiter=','))
            except ValueError as e:
                raise (ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        elif self.class_n is not None:
            self.n_classes = self._parse(self.class_n, int, 'class input should be .csv OR int(num_classes). {}')
            self.classes = {}
            for i in range(self.n_classes):
                self.classes['c'+str(i)] = i
        else:
            raise (ValueError('invalid class input: should be csv file OR int(num_classes). {} {}'.format(self.class_list, e)))


        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        if filter_ext is not None:
            file_names = [fn for fn in self.image_names if (fn[-3:].lower() in filter_ext)]
            for fn in file_names:
                self.image_names.remove(fn)

        #folders filter:
        dirs = [dn for dn in self.image_names if not os.path.isfile(os.path.join(folder_path, dn))]
        for dir_i in dirs:
            self.image_names.remove(dir_i)

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise(ValueError(fmt.format(e)))

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_names[idx])
        try:
            img = self.load_image_path(img_path)
        except:
            raise TypeError('Can not open image \'{}\'. if it is not an image add it to the \'exclude_ext\''.format(img_path))

        annot = np.zeros((0, 5))
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image_path(self, img_path):

        img = skimage.io.imread(img_path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1
    def image_aspect_ratio(self, image_index):
        return 1 # Batch size=1 so sampler is not relevant

