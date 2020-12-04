""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import cv2

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
from PIL import Image

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

class TinyImageNet(Dataset):
  def __init__(self, train_or_test='train', dir=None, preload=True, transform=None):
    metatrain_folder = '%s/train' % dir
    metatest_folder = '%s/val' % dir
    self.num_classes = 200
    self.train_or_test = train_or_test
    self.preload = preload

    self.labels = os.listdir(metatrain_folder)
    assert len(self.labels) == self.num_classes
    self.labels_map = dict(zip(self.labels, range(len(self.labels))))

    if train_or_test == 'train':
      folders = [os.path.join(metatrain_folder, label) \
                  for label in os.listdir(metatrain_folder) \
                    if os.path.isdir(os.path.join(metatrain_folder, label))]
      images = []
      for folder in folders:
        for img_name in os.listdir(os.path.join(folder, 'images')):
          _l = self.labels_map[os.path.basename(folder)]
          images.append([os.path.join(folder, 'images', img_name), _l])
          # images.append([os.path.join(folder, img_name), self.labels_map[os.path.basename(folder)]])
      assert len(images) == 200 * 500
    elif train_or_test == 'test':
      anno_file = os.path.join(metatest_folder, 'val_annotations.txt')
      images = []
      with open(anno_file, 'r') as f:
        for l in f.readlines():
          img_name, label = l.strip().split()[:2]
          images.append([os.path.join(metatest_folder, 'images', img_name), self.labels_map[label]])
      assert len(images) == 200 * 50
    else:
      assert False
    self.data = self.get_images(images)
    self.transform = transform

  def get_images(self, images):
    for i, (img_path, _) in enumerate(images):
      images[i][0] = Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    return images

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    img = self.data[index][0]
    if self.transform is not None:
      img = self.transform(img)
    return img, self.data[index][1]

