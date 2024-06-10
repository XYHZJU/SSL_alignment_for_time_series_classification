from sympy import O
from exp.exp_basic import Exp_Basic

from models import lstm,lstm_pred,lstm_joint
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


class Exp_JOINT_EVAL(Exp_Basic):
    def __init__(self, args):
        super(Exp_JOINT_EVAL, self).__init__(args)
        self.train_scaler = None
        self.args = args
        # self.scaler = calc_data_scale(args)
        self.train_loaderlist = None
        self.valid_loader = None
        self.test_loader = None
        self.step = 0
        self.embedding = None
        self.embedding_average = None
        self.valid_embedding = None
        self.valid_labels = None
        self.labels = None
        self.supcon_loss = SupConLoss(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.mean,self.std = calc_data_scale()

    
    def _build_model(self):
        model_dict = {
            'LSTM':lstm_joint,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    
    def _get_dataloader(self,args,split,shuffle):
        if self.args.dataset == "HAR":
            return create_HAR_ssl_loader(args,split,shuffle)
        else:
            return create_non_graph_ssl_loader(args,split,shuffle)

    def _get_downstream_dataloader(self,args,split,shuffle):
        if self.args.dataset == "HAR":
            return create_HAR_loader(args,split,shuffle,"Classification")
        else:
            return create_non_graph_loader(args,split,shuffle,"Classification")
        

    def _select_optimizer(self, nettype='all'):
        if nettype == 'all':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('wrong type.')
        return model_optim
    
    def adding_noise(self,x):
        noise = torch.randn(x.shape)*self.args.aug_variance
        return x + noise.to(self.device)

    def cluster_loss(self, embedding, train_labels,update_average = False):
        cos_loss = nn.CosineSimilarity(dim=1)
        embedding_dict = torch.zeros(self.args.n_classes, self.args.hidden_dim).to(self.device)
        for label_id in range(self.args.n_classes):
            index_i = [i for i, item in enumerate(train_labels) if item == label_id]
            if len(index_i) == 0:
                continue
            # print("indexes and label: ",label_id, index_i )
            
            embedding_i = embedding[index_i]
            average_batch_embedding_i = torch.mean(embedding_i,dim=0)
            # print("average label i: ",average_batch_embedding_i)
            embedding_dict[label_id] = average_batch_embedding_i
        # embedding_dict.to(self.device)
        if update_average:
            self.embedding_average = embedding_dict
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

        diverge_loss = 0

        # diverge_loss_list = []

        for i in range(embedding_dict.shape[0]):
            for j in range(i,embedding_dict.shape[0]):
                if i==j:
                    continue
                # print(i,j)
                loss = utils.compute_regression_loss(y_true=embedding_dict[i],y_predicted=embedding_dict[j],loss_fn="MAE",standard_scaler=None,device=self.device )
                if loss > self.args.cluster_margin:
                    continue
                diverge_loss += -loss
        return cluster_loss, diverge_loss
    
    @profile
    def train_epoch_batched(self, model, optimizer, train_loader, epoch):
        model.train()
        loss_sup = []
        loss_ssl = []
        loss_joint = []
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.n_epochs)
        cos_loss = nn.CosineSimilarity(dim=1)
        cos_loss2 = nn.CosineSimilarity(dim=0)
        
        for batch_X, batch_y, x_labels in tqdm(train_loader, total=len(train_loader)):

            train_X = batch_X.float().to(self.device)
            train_labels = x_labels.to(self.device)
            # --------------- forward --------------- #
            output_y,predict_class,embedding = model.forward(train_X)
            loss_class = self.criterion(predict_class, train_labels.squeeze().to(self.device))

            if epoch >= self.args.drop_task_epoch:
                loss_total = loss_class
                loss_batch = loss_total*0

            elif self.args.contrastive:
                augment_X = self.adding_noise(train_X)
                _,_,embedding_augment = model.forward(augment_X)
                features = torch.cat([embedding.unsqueeze(1), embedding_augment.unsqueeze(1)], dim=1)
                if self.args.supcon:
                    loss_batch = self.supcon_loss(features,train_labels)
                else:
                    loss_batch = self.supcon_loss(features)
                
                loss_total = loss_batch*self.args.w_auxiliary_task + loss_class*self.args.w_main_task
                
            else:    

                loss_batch = utils.compute_regression_loss(y_true=batch_y.to(self.device),y_predicted=output_y,loss_fn="MAE",standard_scaler=None,device=self.device )
                
                cluster_loss, diverge_loss = self.cluster_loss(embedding, train_labels, update_average=True)
                    
                loss_batch = loss_batch*self.args.cluster_prediction_weight + cluster_loss* self.args.cluster_attract_weight + diverge_loss*self.args.cluster_repel_weight/comb(self.args.n_classes,2)
                # print(predict_class.dtype,train_labels.dtype)
                
                
                loss_total = loss_batch*self.args.w_auxiliary_task + loss_class*self.args.w_main_task



            # ----------- Parameters update --------------- #
            optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.max_grad_norm)
            optimizer.step()
            self.step+=self.args.batch_size
            loss_joint.append(loss_total.item())
            loss_sup.append(loss_class.item())
            loss_ssl.append(loss_batch.item())
            # print("time infer: ",t1)
        loss_sup_ = np.array(loss_sup).mean(axis=0)
        loss_ssl_ = np.array(loss_ssl).mean(axis=0)
        loss_joint_ = np.array(loss_joint).mean(axis=0)
        scheduler.step()

        return loss_joint_, loss_sup_, loss_ssl_

    def ds_inference(self, model, train_loader):
        model.eval()
        self.embedding = None
        self.labels = None
        for batch_X, batch_y in tqdm(train_loader, total=len(train_loader)):
            
            _,_,embedding = model.forward(batch_X.float().to(self.device))
            # print("out shape: ",batch_y.shape,output_y.shape)
            # print("out shape: ",output_y.dtype)

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

        criterion = self.criterion
        loss_list = []
        y_pred_all = []
        y_true_all = []
        y_prob_all = []
        self.valid_embedding = None
        self.valid_labels = None

        for batch_X, batch_y in tqdm(valid_loader, total=len(valid_loader)):
            with torch.no_grad():
                _, output_y, embedding = model.forward(batch_X.float().to(self.device))
            # print(output_y,batch_y)
            y_prob = F.softmax(output_y, dim=1).cpu().numpy()
                # print("y prob: ",y_prob)
            y_pred = np.argmax(y_prob, axis=1).reshape(-1)
            # y_true = batch_y
                
            # if self.args.dataset == "HAR" or self.args.dataset == "EpilepsySmall":
            loss = criterion(output_y, batch_y.squeeze().to(self.device))
            # else:
            #     loss = criterion(output_y, batch_y.to(self.device))
            
            if self.valid_embedding is None:
                self.valid_embedding = embedding.cpu().detach().numpy()
                self.valid_labels = batch_y.numpy().squeeze()
            else:
                self.valid_embedding = np.concatenate((self.valid_embedding,embedding.cpu().detach().numpy()),axis=0)
                self.valid_labels = np.concatenate((self.valid_labels,batch_y.numpy().squeeze()),axis=0)

            # print("current loss: ",loss)
            # print(np.concatenate((np.expand_dims(y_pred,1),np.expand_dims(batch_y,1)),axis=1))
            total_loss += loss.item()
            loss_list.append(loss.item())
            # print("y shape: ",y_pred.shape)
            y_pred_all.append(y_pred)
            y_true_all.append(batch_y)
            y_prob_all.append(y_prob)

        loss_list = np.array(loss_list)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        # print("loss info: ",np.mean(loss_list),np.var(loss_list),len(loss_list),len(valid_loader))
        loss_mean = total_loss/len(loss_list)
        matrix = confusion_matrix(y_true_all,y_pred_all)
        print("confusion_matrix: ")
        print(matrix)

        scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all)

        results_list = [('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision'])]

        return results_list[1][1],loss_mean
    
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

        best_score = -np.inf
        best_epoch, stop_round = 0, 0
        train_joint_loss = []
        train_sup_loss = []
        train_ssl_loss = []

        valid_score_list = []
        test_score_list = []


        valid_loss_list = []
        test_loss_list = []
        train_loss_list = []
        Log_ME_list = []
        SFDA_list = []
        NLEEP_list = []

            
        self.train_loader = self._get_dataloader(self.args,"train",shuffle=True)
        self.ds_train_loader = self._get_downstream_dataloader(self.args,"train",shuffle=False)
        self.ds_vali_loader = self._get_downstream_dataloader(self.args,"valid",shuffle=False)
        self.ds_test_loader = self._get_downstream_dataloader(self.args,"test",shuffle=False)
        

        for epoch in range(1,self.args.n_epochs):
            self.pprint('Epoch:', epoch)
            self.pprint('training...')

            train_loader = self.train_loader
    
            loss_joint, loss_sup, loss_ssl = self.train_epoch_batched(self.model,optimizer, train_loader, epoch)
            self.ds_inference(self.model,self.ds_train_loader)

            self.pprint(loss_joint, "supervised: ", loss_sup, "ssl: ", loss_ssl)
            # self.pprint("embedding average: ",self.embedding_average)

            self.pprint('evaluating...')
            train_loss = None
            valid_loss = None
            test_loss = None

            train_joint_loss.append(loss_joint)
            train_sup_loss.append(loss_sup)
            train_ssl_loss.append(loss_ssl)

            # train_loss = self.valid_batch(self.model, self.train_loader)
            
            
            # train_loss_list.append(train_loss)
            
            
            valid_score, valid_loss = self.valid_batch(self.model, self.ds_vali_loader)
            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting valid embeddings...')
                self.plot_embedding(self.valid_embedding,self.valid_labels,epoch,train_loss,folder_path+'valid_plot/')

            test_score, test_loss = self.valid_batch(self.model, self.ds_test_loader)
            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting test embeddings...')
                self.plot_embedding(self.valid_embedding,self.valid_labels,epoch,train_loss,folder_path+'test_plot/')
            
            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting embeddings...')
                self.plot_embedding(self.embedding,self.labels,epoch,train_loss,folder_path,(valid_loss,test_loss),embedding_dict = self.embedding_average.cpu().detach().numpy())
            
            valid_loss_list.append(valid_loss)
            test_loss_list.append(test_loss)
            
            valid_score_list.append(valid_score)
            test_score_list.append(test_score)

        
            logme_value = self.LogME_basic(self.embedding,self.labels)
            sfda_value = self.SFDA_score(self.embedding,self.labels)
            nleep_value = self.NLEEP_score(self.embedding,self.labels)
            Log_ME_list.append(logme_value)
            SFDA_list.append(sfda_value)
            NLEEP_list.append(nleep_value)

            self.pprint('valid %.6f, test %.6f, logme %.6f, sfda %.6f, nleep %.6f' %
            (valid_score, test_score, logme_value, sfda_value, nleep_value))

            if valid_score > best_score:
                best_score = valid_score
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
        self.plot_joint_loss(train_joint_loss,train_sup_loss,train_ssl_loss,folder_path,self.args)
        self.plot_score(train_loss_list,valid_score_list,test_score_list,folder_path,self.args)
        self.plot_loss(train_sup_loss,valid_loss_list,test_loss_list,folder_path,self.args)
        self.plot_SSL_metrics(Log_ME_list,SFDA_list,NLEEP_list, folder_path,self.args)
        print("train data path: ",str(folder_path))

        return self.model
    
    def test(self, settings):
        output_path = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.args.log_file = os.path.join(output_path, 'run.log')
        
        if not self.args.is_training:
            self.ds_test_loader = self._get_downstream_dataloader(self.args,"test",shuffle=False)
        print("model path: ",str(self.args.log_file))
        criterion = self.criterion


        self.pprint('load models...')
        self.model.load_state_dict(torch.load(os.path.join(output_path, 'checkpoint.pth'), map_location='cpu'))
        y_pred_all = []
        y_true_all = []
        y_prob_all = []
        loss_list = []
        total_loss = 0
        
        self.pprint('Calculate the metrics.')
        self.model.eval()
        with torch.no_grad():

            for batch_X, batch_y in tqdm(self.ds_test_loader, total=len(self.ds_test_loader)):
                _,output_y,_ = self.model.forward(batch_X.float().to(self.device))
                # print("y prob: ",output_y)
                # print("batch_y shape: ",batch_y.shape)
                y_prob = F.softmax(output_y, dim=1).cpu().numpy()
                # print("y prob: ",y_prob)
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)
                y_true = batch_y
                # print(y_pred,y_true)
                # print(torch.unique(batch_y))
                # if self.args.dataset == "HAR" or self.args.dataset == "EpilepsySmall":
                loss = criterion(output_y, y_true.squeeze().to(self.device))
                # else:
                #     loss = criterion(output_y, y_true.to(self.device))
            
                total_loss += loss.item()
                loss_list.append(loss.item())
                # print("y_pred shape: ",y_pred.shape)

                y_pred_all.append(y_pred)
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)

            y_pred_all = np.concatenate(y_pred_all, axis=0)
            y_true_all = np.concatenate(y_true_all, axis=0)
            y_prob_all = np.concatenate(y_prob_all, axis=0)
        print("len y: ",y_pred_all.shape[0])
        unique_elements, counts = np.unique(y_pred_all, return_counts=True)
        element_counts = dict(zip(unique_elements, counts))
        for element, count in element_counts.items():
            print(f"class {element} occur {count} times")
        loss_list = np.array(loss_list)
        loss_mean = total_loss/len(loss_list)
        matrix = confusion_matrix(y_true_all,y_pred_all)
        print("confusion_matrix: ")
        print(matrix)

        scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all)

        results_list = [('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision'])]

        self.pprint('the result of the test set:')
        self.pprint('acc:{}, F1:{}, recall:{}, precision: {}, cross_entropy: {}'.format(results_list[0][1],results_list[1][1],results_list[2][1],results_list[3][1],loss_mean))
        

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
        

    