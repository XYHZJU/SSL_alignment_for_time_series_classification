import numpy as np
import pandas as pd
import torch
from data.Dataset import TestDataset_HAR,TestDataset_SSL_HAR
from torch.utils.data import Dataset, DataLoader

import time
import os

import psutil
import scipy.signal as signal
import random






def create_non_graph_loader(args,split,shuffle,task_op = None):
    seed = args.seed
    torch.manual_seed(seed)
    if task_op is not None:
        TASK = task_op
    else:
        TASK = args.task
    dataset = args.dataset
    
    if TASK == "Classification":
        if split == "train":
            x = np.load('src/data/'+dataset+'/Classification/train_x.npy')
            y = np.load('src/data/'+dataset+'/Classification/train_x_labels.npy')
        elif split == "valid":
            x = np.load('src/data/'+dataset+'/Classification/vali_x.npy')
            y = np.load('src/data/'+dataset+'/Classification/vali_x_labels.npy')
        else:
            x = np.load('src/data/'+dataset+'/Classification/test_x.npy')
            y = np.load('src/data/'+dataset+'/Classification/test_x_labels.npy')

    else:
        if split == "train":
            x = np.load('src/data/'+dataset+'/Detection/train_x.npy')
            y = np.load('src/data/'+dataset+'/Detection/train_x_labels.npy')
        elif split == "valid":
            x = np.load('src/data/'+dataset+'/Detection/vali_x.npy')
            y = np.load('src/data/'+dataset+'/Detection/vali_x_labels.npy')
        else:
            x = np.load('src/data/'+dataset+'/Detection/test_x.npy')
            y = np.load('src/data/'+dataset+'/Detection/test_x_labels.npy')

    
    print("create ",split," set: ",x.shape[0])

    dataset = TestDataset_HAR(x,y,args,split)
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator = g)

def create_non_graph_ssl_loader(args,split,shuffle = False):
    seed = args.seed
    torch.manual_seed(seed)

    dataset = args.dataset

    
    if split == "train":

        datasetx = np.load('src/data/'+dataset+'/SSL/train_x.npy')
        datasety = np.load('src/data/'+dataset+'/SSL/train_y.npy')
        dataset_label = np.load('src/data/'+dataset+'/SSL/train_x_labels.npy')
    elif split == "valid":
        datasetx = np.load('src/data/'+dataset+'/SSL/vali_x.npy')
        datasety = np.load('src/data/'+dataset+'/SSL/vali_y.npy')
        dataset_label = np.load('src/data/'+dataset+'/SSL/vali_x_labels.npy')
    else:
        datasetx = np.load('src/data/'+dataset+'/SSL/test_x.npy')
        datasety = np.load('src/data/'+dataset+'/SSL/test_y.npy')
        dataset_label = np.load('src/data/'+dataset+'/SSL/test_x_labels.npy')

    print("creating loader for: ",split,"len: ",len(datasetx))    

    dataset = TestDataset_SSL_HAR(datasetx,dataset_label,datasety,args,split)
    print("create finished")
    g = torch.Generator()
    g.manual_seed(seed)
    
    return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator=g)

def create_HAR_loader(args,split,shuffle,task_op = None):
    seed = args.seed
    torch.manual_seed(seed)
    if task_op is not None:
        TASK = task_op
    else:
        TASK = args.task
    
    if TASK == "Classification":
        if split == "train":
            x = np.load('src/data/HAR/Classification/train_x.npy')
            y = np.load('src/data/HAR/Classification/train_x_labels.npy')
        elif split == "valid":
            x = np.load('src/data/HAR/Classification/vali_x.npy')
            y = np.load('src/data/HAR/Classification/vali_x_labels.npy')
        else:
            x = np.load('src/data/HAR/Classification/test_x.npy')
            y = np.load('src/data/HAR/Classification/test_x_labels.npy')
    else:
        if split == "train":
            x = np.load('src/data/HAR/Detection/train_x.npy')
            y = np.load('src/data/HAR/Detection/train_x_labels.npy')
        elif split == "valid":
            x = np.load('src/data/HAR/Detection/vali_x.npy')
            y = np.load('src/data/HAR/Detection/vali_x_labels.npy')
        else:
            x = np.load('src/data/HAR/Detection/test_x.npy')
            y = np.load('src/data/HAR/Detection/test_x_labels.npy')
    
    print("create ",split," set: ",x.shape[0])

    dataset = TestDataset_HAR(x,y,args,split)
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator = g)

def create_HAR_ssl_loader(args,split,shuffle = False):
    seed = args.seed
    torch.manual_seed(seed)

    
    if split == "train":

        datasetx = np.load('src/data/HAR/SSL/train_x.npy')
        datasety = np.load('src/data/HAR/SSL/train_y.npy')
        dataset_label = np.load('src/data/HAR/SSL/train_x_labels.npy')
    elif split == "valid":
        datasetx = np.load('src/data/HAR/SSL/vali_x.npy')
        datasety = np.load('src/data/HAR/SSL/vali_y.npy')
        dataset_label = np.load('src/data/HAR/SSL/vali_x_labels.npy')
    else:
        datasetx = np.load('src/data/HAR/SSL/test_x.npy')
        datasety = np.load('src/data/HAR/SSL/test_y.npy')
        dataset_label = np.load('src/data/HAR/SSL/test_x_labels.npy')

    print("creating loader for: ",split,"len: ",len(datasetx))    

    dataset = TestDataset_SSL_HAR(datasetx,dataset_label,datasety,args,split)
    print("create finished")
    g = torch.Generator()
    g.manual_seed(seed)
    
    return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator=g)

def create_HAR70_loader(args,split,shuffle,task_op = None):
    seed = args.seed
    torch.manual_seed(seed)
    if task_op is not None:
        TASK = task_op
    else:
        TASK = args.task
    

    if split == "train":
        x = np.load('src/data/HAR70plus/Classification/train_x.npy')
        y = np.load('src/data/HAR70plus/Classification/train_x_labels.npy')
    elif split == "valid":
        x = np.load('src/data/HAR70plus/Classification/vali_x.npy',allow_pickle = True)
        y = np.load('src/data/HAR70plus/Classification/vali_x_labels.npy',allow_pickle = True)
    else:
        x = np.load('src/data/HAR70plus/Classification/test_x.npy',allow_pickle = True)
        y = np.load('src/data/HAR70plus/Classification/test_x_labels.npy',allow_pickle = True)
    
    print("create ",split," set: ",x.shape[0])

    dataset = TestDataset_HAR(x,y,args,split)
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator = g)

def create_HAR70_ssl_loader(args,split,shuffle = False):
    seed = args.seed
    torch.manual_seed(seed)

    
    if split == "train":

        datasetx = np.load('src/data/HAR70plus/SSL/train_x.npy')
        datasety = np.load('src/data/HAR70plus/SSL/train_y.npy')
        dataset_label = np.load('src/data/HAR70plus/SSL/train_x_labels.npy')
    elif split == "valid":
        datasetx = np.load('src/data/HAR70plus/SSL/vali_x.npy')
        datasety = np.load('src/data/HAR70plus/SSL/vali_y.npy')
        dataset_label = np.load('src/data/HAR70plus/SSL/vali_x_labels.npy')
    else:
        datasetx = np.load('src/data/HAR70plus/SSL/test_x.npy')
        datasety = np.load('src/data/HAR70plus/SSL/test_y.npy')
        dataset_label = np.load('src/data/HAR70plus/SSL/test_x_labels.npy')

    print("creating loader for: ",split,"len: ",len(datasetx))    

    dataset = TestDataset_SSL_HAR(datasetx,dataset_label,datasety,args,split)
    print("create finished")
    g = torch.Generator()
    g.manual_seed(seed)
    
    return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator=g)

