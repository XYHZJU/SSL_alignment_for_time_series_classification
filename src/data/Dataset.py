from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal
import numpy as np
import pickle
import torch
import time
import utils
import pandas as pd

from line_profiler import profile


class TestDataset_HAR(Dataset):
    def __init__(self, x,y,args,split):
        """
        Args:
            data (list)
            labels (list)
            transform (callable, optional)
        """
        self.x = x
        self.y = y
        self.split = split
        self.args = args

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # t0 = time.time()
        current_xi = self.x[idx]
        if self.args.dataset == "HAR":
            channels,length = current_xi.shape
        else:
            length,channels = current_xi.shape
        # print(current_xi.shape)
        current_xi = current_xi.reshape(self.args.max_clip_length,-1,channels)

        # current_xi = torch.Tensor(current_xi)

        y = self.y[idx]
        # print("task: ",self.args.task)
        if self.args.task == "Classification" or self.args.task == "SSLJoint":
            y = torch.LongTensor([y])
        else:
            y = torch.FloatTensor([y])

        return current_xi,y

class TestDataset_SSL_HAR(Dataset):
    def __init__(self, x,x_labels,y,args,split):
        """
        Args:
            data (list)
            labels (list)
            transform (callable, optional)
        """
        self.x = x
        self.y = y
        self.x_labels = x_labels
        self.split = split
        self.args = args

    def __len__(self):
        return self.y.shape[0]
        
    def __getitem__(self, idx):
        # t0 = time.time()
        current_xi = self.x[idx]
        current_yi = self.y[idx]
        if self.args.dataset == "HAR":
            channels,length = current_xi.shape
        else:
            length,channels = current_xi.shape
        # channels,length = current_xi.shape
        current_xi = current_xi.reshape(self.args.max_clip_length,-1,channels)
        current_yi = current_yi.reshape(self.args.max_clip_length,-1,channels)

        current_xi = torch.Tensor(current_xi)
        current_yi = torch.Tensor(current_yi)

        x_labels = self.x_labels[idx]
        if self.args.task == "SSLJointDetection":
            x_labels = torch.FloatTensor([x_labels])
        else:
            x_labels = torch.LongTensor([x_labels])
        


        return current_xi,current_yi,x_labels










