from turtle import update
from exp.exp_basic import Exp_Basic

from models import lstm,lstm_pred
from models.supcon import SupConLoss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import os
import datetime
import gc
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter
import time
from line_profiler import profile
from scipy.special import comb

from data.dataloader import create_HAR_ssl_loader,create_HAR_loader,create_non_graph_loader,create_non_graph_ssl_loader
import utils




warnings.filterwarnings('ignore')


class Exp_SSL_EVAL(Exp_Basic):
    def __init__(self, args):
        super(Exp_SSL_EVAL, self).__init__(args)
        self.train_scaler = None
        self.args = args
        # self.scaler = calc_data_scale(args)
        self.train_loaderlist = None
        self.valid_loader = None
        self.test_loader = None
        self.step = 0
        self.embedding = None
        self.embedding_average = None
        self.mu_x = None
        self.wx = None
        self.wx_labels = None
        self.batch_average = []
        self.labels = None
        self.supcon_loss = SupConLoss(self.device)
        self.cluster_loss_val = None
        self.diverge_loss_val = None
        # self.mean,self.std = calc_data_scale()

    
    def _build_model(self):
        model_dict = {
            'LSTM':lstm_pred,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    
    def _get_dataloader(self,args,split,shuffle):
        
        return create_non_graph_ssl_loader(args,split,shuffle)

    def _get_downstream_dataloader(self,args,split,shuffle):

        task_op = "Classification"
        if self.args.dataset == "Heartbeat":
            task_op = "Detection"
            
        
        return create_non_graph_loader(args,split,shuffle,task_op)

    def _select_optimizer(self, nettype='all'):
        if nettype == 'all':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('wrong type.')
        return model_optim
    
    def adding_noise(self,x):
        noise = torch.randn(x.shape)*self.args.aug_variance
        return x + noise.to(self.device)
    
    def mu_embedding(self,y_true,y_label,y_predict):
        n = y_true.shape[0]
        y_true = y_true.reshape((n,-1))
        # y_label = y_label.reshape((n,-1))
        y_predict = y_predict.reshape((n,-1))
        mu_x = np.zeros((self.args.n_classes, y_true.shape[-1]))
        for label_id in range(self.args.n_classes):
            index_i = [i for i, item in enumerate(y_label) if item == label_id]
            if len(index_i) == 0:
                continue
            # print("indexes and label: ",label_id, index_i )
            embedding_i = y_true[index_i]
            mu_x[label_id] = np.mean(embedding_i,axis=0)
        self.mu_x = mu_x
        self.wx = y_predict
        self.wx_labels = y_label

    
    def cluster_loss(self, model, embedding, train_labels,update_average = False):
        cos_loss = nn.CosineSimilarity(dim=1)
        embedding_dict = torch.zeros(self.args.n_classes, self.args.hidden_dim).to(self.device)
        # print(model.seen_labels)
        for label_id in range(self.args.n_classes):
            index_i = [i for i, item in enumerate(train_labels) if item == label_id]
            if len(index_i) == 0:
                continue
            # print("indexes and label: ",label_id, index_i )
            embedding_i = embedding[index_i]
            average_batch_embedding_i = torch.mean(embedding_i,dim=0)
            num_label_i = len(index_i)
            # print("average label i: ",average_batch_embedding_i)
            # print(label_id, model.history_embedding.shape, len(model.seen_labels))
            # embedding_dict[label_id] = (average_batch_embedding_i * 0.9 + model.history_embedding[label_id] * 0.1)
            embedding_dict[label_id] = average_batch_embedding_i
            # model.seen_labels[label_id] = model.seen_labels[label_id]+num_label_i
            
            model.history_embedding[label_id] = (embedding_dict[label_id].detach() * num_label_i + model.history_embedding[label_id] * model.seen_labels[label_id])/(model.seen_labels[label_id]+num_label_i)
            model.seen_labels[label_id] = model.seen_labels[label_id]+num_label_i
        # embedding_dict.to(self.device)
        if update_average:
            self.embedding_average = model.history_embedding
            self.batch_average.append(embedding_dict)
        embedding_targets = []
        embedding_opposites = []
        for i in range(train_labels.shape[0]):
            current_label = train_labels[i]
            # print(current_label, embedding_dict)
            embedding_target = embedding_dict[current_label]

            indices = [i for i in range(embedding_dict.shape[0]) if i!=current_label.item()]
            # print(current_label, indices)
            embedding_opposite = embedding_dict[indices]

            embedding_targets.append(embedding_target)
            embedding_opposites.append(embedding_opposite)

        embedding_targets = torch.cat(embedding_targets,dim=0)
        embedding_opposites = torch.cat(embedding_opposites,dim=0).view(train_labels.shape[0],embedding_dict.shape[0]-1,embedding_dict.shape[1])

        cluster_loss = -cos_loss(embedding, embedding_targets).mean()
        # cluster_loss = utils.compute_regression_loss(y_true=embedding_targets,y_predicted=embedding,loss_fn="MAE",standard_scaler=None,device=self.device )

        diverge_loss = 0

        # diverge_loss_list = []

        for i in range(embedding_dict.shape[0]):
            for j in range(i,embedding_dict.shape[0]):
                if i==j:
                    continue
                # print(i,j)
                loss = utils.compute_regression_loss(y_true=embedding_dict[i],y_predicted=embedding_dict[j],loss_fn="MAE",standard_scaler=None,device=self.device )
                # print(i,j,loss)
                if loss > self.args.cluster_margin:
                    continue
                diverge_loss += -loss
        return cluster_loss, diverge_loss
    
    @profile
    def train_epoch_batched(self, model, optimizer, train_loader, epoch):
        model.train()
        loss_sup = []
        loss_cluster = []
        loss_diverge = []
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.n_epochs)
        cos_loss2 = nn.CosineSimilarity(dim=0)

        if self.args.cluster:
            model.history_embedding = torch.zeros(self.args.n_classes, self.args.hidden_dim).to(self.device)
            model.history_embedding.requires_grad = False
            model.seen_labels = [0]*self.args.n_classes
            self.batch_average = []

        for batch_X, batch_y, x_labels in tqdm(train_loader, total=len(train_loader)):

            train_X = batch_X.float().to(self.device)
            train_labels = x_labels.to(self.device)
            # --------------- forward --------------- #
            output_y,embedding = model.forward(train_X)
            # print(output_y.shape, batch_y.shape)
            loss_batch = utils.compute_regression_loss(y_true=batch_y.to(self.device),y_predicted=output_y,loss_fn="MAE",standard_scaler=None,device=self.device )

            loss_batch = loss_batch

            if self.args.contrastive:
                augment_X = self.adding_noise(train_X)
                _,embedding_augment = model.forward(augment_X)
                features = torch.cat([embedding.unsqueeze(1), embedding_augment.unsqueeze(1)], dim=1)
                if self.args.supcon:
                    loss_batch = self.supcon_loss(features,train_labels)
                else:
                    loss_batch = self.supcon_loss(features)


            if self.args.cluster:
                cluster_loss, diverge_loss = self.cluster_loss(model, embedding, train_labels, update_average=True)
                
                loss_batch = loss_batch*self.args.cluster_prediction_weight + cluster_loss* self.args.cluster_attract_weight + diverge_loss*self.args.cluster_repel_weight/comb(self.args.n_classes,2)
                
                # print("loss: ",cluster_loss, diverge_loss)
                loss_cluster.append(cluster_loss.item())
                try:
                    loss_diverge.append(diverge_loss.item())
                except:
                    loss_diverge.append(diverge_loss)
                loss_cluster_ = np.array(loss_cluster).mean(axis=0)
                loss_diverge_ = np.array(loss_diverge).mean(axis=0)

                self.cluster_loss_val = loss_cluster_
                self.diverge_loss_val = loss_diverge_

            # ----------- Parameters update --------------- #
            optimizer.zero_grad()
            loss_batch.backward()
            nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.max_grad_norm)
            optimizer.step()
            self.step+=self.args.batch_size
            loss_sup.append(loss_batch.item())
            
            # print("time infer: ",t1)
        loss_sup_ = np.array(loss_sup).mean(axis=0)
        scheduler.step()

        return loss_sup_

    def ds_inference(self, model, train_loader):
        model.eval()
        self.embedding = None
        self.labels = None
        if self.args.cluster:
            model.history_embedding = torch.zeros(self.args.n_classes, self.args.hidden_dim).to(self.device)
            model.history_embedding.requires_grad = False
            model.seen_labels = [0]*self.args.n_classes

        for batch_X, batch_y in tqdm(train_loader, total=len(train_loader)):
            
            _,embedding = model.forward(batch_X.float().to(self.device))
            # print("out shape: ",batch_y.shape,output_y.shape)
            # print("out shape: ",output_y.dtype
            embedding = embedding.cpu().detach().numpy()

            if self.embedding is None:
                self.embedding = embedding
                self.labels = batch_y.cpu().detach().numpy().squeeze()
            else:
                self.embedding = np.concatenate((self.embedding,embedding),axis=0)
                self.labels = np.concatenate((self.labels,batch_y.cpu().detach().numpy().squeeze()),axis=0)
            # ----------- Parameters update --------------- #

    
    def valid_batch(self, model, valid_loader):
        model.eval()
        total_loss = 0

        
        loss_list = []
        y_pred_all = []
        y_true_all = []

        y_true_embedding = []
        y_labels = []
        y_predict_embedding = []

        if self.args.cluster:
            model.history_embedding = torch.zeros(self.args.n_classes, self.args.hidden_dim).to(self.device)
            model.history_embedding.requires_grad = False
            model.seen_labels = [0]*self.args.n_classes

        for batch_X, batch_y, x_labels in tqdm(valid_loader, total=len(valid_loader)):
            output_y,embedding = model.forward(batch_X.float().to(self.device))
            train_labels = x_labels.to(self.device)

            y_true_embedding.append(batch_y[:-1].cpu().detach().numpy())
            y_labels.append(x_labels[1:].cpu().detach().numpy())
            y_predict_embedding.append(output_y[:-1].cpu().detach().numpy())

            # print(output_y,batch_y)
            loss = utils.compute_regression_loss(y_true=batch_y.to(self.device),y_predicted=output_y,loss_fn="MAE",standard_scaler=None,device=self.device )
            
            if self.args.contrastive:
                augment_X = self.adding_noise(batch_X.float().to(self.device))
                _,embedding_augment = model.forward(augment_X)
                features = torch.cat([embedding.unsqueeze(1), embedding_augment.unsqueeze(1)], dim=1)
                if self.args.supcon:
                    loss = self.supcon_loss(features,train_labels)
                else:
                    loss = self.supcon_loss(features)

            if self.args.cluster:
                cluster_loss, diverge_loss = self.cluster_loss(model, embedding, train_labels)
                loss = loss*self.args.cluster_prediction_weight + cluster_loss* self.args.cluster_attract_weight + diverge_loss*self.args.cluster_repel_weight/comb(self.args.n_classes,2)
            
            total_loss += loss.item()
            loss_list.append(loss.item())
            # print("y shape: ",y_pred.shape)
            y_pred_all.append(output_y.cpu().detach().numpy())
            y_true_all.append(batch_y.cpu().detach().numpy())

        loss_list = np.array(loss_list)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_true_all = np.concatenate(y_true_all, axis=0)

        y_true_embedding = np.concatenate(y_true_embedding,axis=0)
        y_labels = np.concatenate(y_labels,axis=0)
        y_predict_embedding = np.concatenate(y_predict_embedding,axis = 0)
        self.mu_embedding(y_true_embedding,y_labels,y_predict_embedding)

        # print("loss info: ",np.mean(loss_list),np.var(loss_list),len(loss_list),len(valid_loader))
        loss_mean = total_loss/len(loss_list)
        return loss_mean
    
    def train(self, settings):
        print("ssl pretrain:")
        output_path = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        folder_path = './train_loss/' + settings + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(folder_path+'npys/'):
            os.makedirs(folder_path+'npys/')
        self.args.log_file = os.path.join(output_path, 'run.log')

        self.pprint("------------------------------------------------------------")
        self.pprint("git branch name: ",self.branch_name)

        for var in vars(self.args):
            self.pprint('{}:{}'.format(var,vars(self.args)[var]))
        
        
        num_model = self.count_parameters(self.model)
        self.pprint('#model params:', num_model)

        optimizer = self._select_optimizer()

        best_score = np.inf
        best_epoch, stop_round = 0, 0
        train_loss_list = []
        valid_loss_list = []
        test_loss_list = []
        train_cluster_loss = []
        train_diverge_loss = []
        Log_ME_list = []
        SFDA_list = []
        NLEEP_list = []

            
        self.vali_loader = self._get_dataloader(self.args,"valid",shuffle=False)
        self.test_loader = self._get_dataloader(self.args,"test",shuffle=False)
        self.train_loader = self._get_dataloader(self.args,"train",shuffle=True)
        self.ds_train_loader = self._get_downstream_dataloader(self.args,"train",shuffle=False)
        

        for epoch in range(1,self.args.n_epochs):
            self.pprint('Epoch:', epoch)
            self.pprint('training...')

            train_loader = self.train_loader
    
            loss_sup = self.train_epoch_batched(self.model,optimizer, train_loader, epoch)
            self.ds_inference(self.model,self.ds_train_loader)

            if self.args.cluster:
                self.pprint(loss_sup,"cluster loss : ", self.cluster_loss_val, "diverge loss :", self.diverge_loss_val)
                train_cluster_loss.append(self.cluster_loss_val)
                train_diverge_loss.append(self.diverge_loss_val)
            else:
                self.pprint(loss_sup)

            self.pprint('evaluating...')
            train_loss = None
            valid_loss = None
            test_loss = None

            train_loss = self.valid_batch(self.model, self.train_loader)
            mu_x = self.mu_x
            wx = self.wx
            wx_labels = self.wx_labels

            valid_loss = self.valid_batch(self.model, self.vali_loader)
            test_loss = self.valid_batch(self.model, self.test_loader)

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            test_loss_list.append(test_loss)

            # print(self.embedding_average)
            # print(self.embedding)

            # print("len batch embedding: ",len(self.batch_average))

            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting embeddings...')
                if self.args.cluster:
                    self.plot_embedding(self.embedding,self.labels,epoch,train_loss,folder_path,(valid_loss,test_loss),embedding_dict = self.embedding_average.cpu().detach().numpy(), embedding_batch = self.batch_average)
                else:
                    self.plot_embedding(self.embedding,self.labels,epoch,train_loss,folder_path,(valid_loss,test_loss))
                self.plot_mu_x_embedding(mu_x,wx,wx_labels,epoch,folder_path)
        
            logme_value = self.LogME_basic(self.embedding,self.labels)
            sfda_value = self.SFDA_score(self.embedding,self.labels)
            nleep_value = self.NLEEP_score(self.embedding,self.labels)
            Log_ME_list.append(logme_value)
            SFDA_list.append(sfda_value)
            NLEEP_list.append(nleep_value)

            self.pprint('valid %.6f, test %.6f, logme %.6f, sfda %.6f, nleep %.6f' %
            (valid_loss, test_loss, logme_value, sfda_value, nleep_value))

            if valid_loss < best_score:
                best_score = valid_loss
                stop_round = 0
                best_epoch = epoch
                torch.save(self.model.state_dict(), output_path+'/checkpoint.pth')
            else:
                stop_round += 1
                if stop_round >= self.args.early_stop:
                    self.pprint('early stop')
                    break
            # adjust_learning_rate(optimizer, epoch+1, self.args)

        self.pprint('best val score:', best_score, '@', best_epoch)
        if self.args.cluster:
            self.plot_cluster_loss(train_cluster_loss,train_diverge_loss,folder_path,self.args)
        self.plot_loss(train_loss_list,valid_loss_list,test_loss_list,folder_path,self.args)
        self.plot_SSL_metrics(Log_ME_list,SFDA_list,NLEEP_list, folder_path,self.args)
        print("train data path: ",str(folder_path))

        return self.model
    
    def test(self, settings):
        
        output_path = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.args.log_file = os.path.join(output_path, 'run.log')
        
        print("model path: ",str(self.args.log_file))

        if not self.args.is_training:
            self.test_loader = self._get_dataloader(self.args,"test",shuffle=False)

        self.pprint('load models...')
        self.model.load_state_dict(torch.load(os.path.join(output_path, 'checkpoint.pth'), map_location='cpu'))
        y_pred_all = []
        y_true_all = []
        loss_list = []
        total_loss = 0
        
        self.pprint('Calculate the metrics.')
        self.model.eval()
        if self.args.cluster:
            self.model.history_embedding = torch.zeros(self.args.n_classes, self.args.hidden_dim).to(self.device)
            self.model.history_embedding.requires_grad = False
            self.model.seen_labels = [0]*self.args.n_classes
        with torch.no_grad():
            
            for batch_X,batch_y, x_labels in tqdm(self.test_loader, total=len(self.test_loader)):

                output_y,embedding = self.model.forward(batch_X.float().to(self.device))
                train_labels = x_labels.to(self.device)
                
                loss = utils.compute_regression_loss(y_true=batch_y.to(self.device),y_predicted=output_y,loss_fn="MAE",standard_scaler=None,device=self.device )
                
                if self.args.contrastive:
                    augment_X = self.adding_noise(batch_X.float().to(self.device))
                    _,embedding_augment = self.model.forward(augment_X)
                    features = torch.cat([embedding.unsqueeze(1), embedding_augment.unsqueeze(1)], dim=1)
                    if self.args.supcon:
                        loss = self.supcon_loss(features,train_labels)
                    else:
                        loss = self.supcon_loss(features)

                if self.args.cluster:
                    cluster_loss, diverge_loss = self.cluster_loss(self.model, embedding, train_labels)
                    loss = loss*self.args.cluster_prediction_weight + cluster_loss* self.args.cluster_attract_weight + diverge_loss*self.args.cluster_repel_weight/comb(self.args.n_classes,2)

                total_loss += loss.item()
                loss_list.append(loss.item())
                # print("y_pred shape: ",y_pred.shape)

                y_pred_all.append(output_y.cpu().detach().numpy())
                y_true_all.append(batch_y.cpu().detach().numpy())
                        
            y_pred_all = np.concatenate(y_pred_all, axis=0)
            y_true_all = np.concatenate(y_true_all, axis=0)

        loss_list = np.array(loss_list)
        loss_mean = total_loss/len(loss_list)

        self.pprint('the result of the test set:',loss_mean)
        
        

        return 
    
    def pprint(self, *text):
        # print with UTC+8 time
        time_ = '['+str(datetime.datetime.utcnow() + 
                        datetime.timedelta(hours=8))[:19]+'] -'
        print(time_, *text, flush=True)
        if self.args.log_file is None:
            return
        with open(self.args.log_file, 'a') as f:
            print(time_, *text, flush=True, file=f)
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        

    