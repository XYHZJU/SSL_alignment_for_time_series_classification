from exp.exp_basic import Exp_Basic

from models import lstm,lstm_pred
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


# from data.dataloader_test import dataloader_test
from data.dataloader import create_non_graph_loader
import utils
import copy




warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.train_scaler = None
        self.criterion = self._select_criterion()
        self.args = args
        if self.args.dataset == "Epilepsy":
            self.scaler = calc_data_scale(args)
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.embedding = None
        self.labels = None
        self.valid_embedding = None
        self.valid_labels = None
        # self.mean,self.std = calc_data_scale()
        if args.fine_tune:
            self.args_pretrained = copy.deepcopy(args)
            # setattr(
            #         self.args_pretrained,
            #         'num_rnn_layers',
            #         args.pretrained_num_rnn_layers)
            self.pretrained_model = self._build_pretrained_model().to(self.device)
            pretrain_output_path = os.path.join(self.args.checkpoints, args.pretrain_model_path)
            self.pretrained_model.load_state_dict(torch.load(os.path.join(pretrain_output_path, 'checkpoint.pth'), map_location='cpu'))
            self.model = utils.build_finetune_lstm_model(model_new=self.model,model_pretrained=self.pretrained_model,num_rnn_layers=args.num_rnn_layers)

    
    def _build_model(self):
        model_dict = {
            'LSTM':lstm
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _build_pretrained_model(self):
        model_dict = {
            'LSTM_Pred' : lstm_pred,
            'LSTM_SIMSIAM':lstm_pred_simsiam

        }
        model = model_dict[self.args.pretrain_model].Model(self.args_pretrained).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    

    def _select_optimizer(self, nettype='all'):
        if nettype == 'all':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('wrong type.')
        return model_optim
    
    def _select_criterion(self):
        # criterion = nn.CrossEntropyLoss()
        if self.args.dataset != "Epilepsy":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = utils.MultiClassFocalLossWithAlpha(self.device)
        return criterion

    
    def train_epoch_batched(self, model, optimizer, train_loader):
        model.train()
        criterion = self.criterion
        loss_sup = []
        self.embedding = None
        self.labels = None
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.n_epochs)
        for batch_X, batch_y in tqdm(train_loader, total=len(train_loader)):
            # --------------- forward --------------- #
            channel = batch_X.reshape(-1,batch_X.shape[-1])
            # print("batch shape:",batch_X.shape,channel.shape)
            # meanc = torch.mean(channel,dim=0)
            # print("mean of each channel: ", meanc)
            output_y,embedding = model.forward(batch_X.float().to(self.device))
            # print("output test: ",output_y.shape,batch_y.to(self.device).shape,type(output_y),type(batch_y.to(self.device)))
            if self.args.dataset != "Epilepsy":
                loss_batch = criterion(output_y, batch_y.squeeze().to(self.device))
            else:
                loss_batch = criterion(output_y, batch_y.to(self.device))
            
            
            if self.embedding is None:
                self.embedding = embedding
                self.labels = batch_y.cpu().detach().numpy().squeeze()
            else:
                self.embedding = np.concatenate((self.embedding,embedding),axis=0)
                self.labels = np.concatenate((self.labels,batch_y.cpu().detach().numpy().squeeze()),axis=0)
            # ----------- Parameters update --------------- #
            optimizer.zero_grad()
            loss_batch.backward()
            nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.max_grad_norm)
            optimizer.step()
            # print(output_y,batch_y)
            # print("current loss: ",loss_batch)
            # y_prob = F.softmax(output_y, dim=1).cpu().detach().numpy()
                # print("y prob: ",y_prob)
            # y_pred = np.argmax(y_prob, axis=1).reshape(-1)
            # print("other class: ",np.sum(y_pred != 0))
            loss_sup.append(loss_batch.item())
        loss_sup_ = np.array(loss_sup).mean(axis=0)
        scheduler.step()

        return loss_sup_
    
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
                output_y,embedding = model.forward(batch_X.float().to(self.device))
            # print(output_y,batch_y)
            y_prob = F.softmax(output_y, dim=1).cpu().numpy()
                # print("y prob: ",y_prob)
            y_pred = np.argmax(y_prob, axis=1).reshape(-1)
            # y_true = batch_y
                
            if self.args.dataset != "Epilepsy":
                loss = criterion(output_y, batch_y.squeeze().to(self.device))
            else:
                loss = criterion(output_y, batch_y.to(self.device))
            
            if self.valid_embedding is None:
                self.valid_embedding = embedding
                self.valid_labels = batch_y.numpy().squeeze()
            else:
                self.valid_embedding = np.concatenate((self.valid_embedding,embedding),axis=0)
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
        output_path = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        folder_path = './train_loss/' + settings + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(folder_path+'npys/'):
            os.makedirs(folder_path+'npys/')
        self.args.log_file = os.path.join(output_path, 'run.log')

        for var in vars(self.args):
            self.pprint('{}:{}'.format(var,vars(self.args)[var]))
        
        
        num_model = self.count_parameters(self.model)
        print('#model params:', num_model)

        optimizer = self._select_optimizer()

        best_score = -np.inf
        best_epoch, stop_round = 0, 0

        train_score_list = []
        valid_score_list = []
        test_score_list = []

        train_loss_list = []
        valid_loss_list = []
        test_loss_list = []
        Log_ME_list = []
        SFDA_list = []
        NLEEP_list = []


        task_op = None
        

        self.train_loader = create_non_graph_loader(self.args,"train",shuffle=True,task_op=task_op)
        self.vali_loader = create_non_graph_loader(self.args,"valid",shuffle=False,task_op=task_op)
        self.test_loader = create_non_graph_loader(self.args,"test",shuffle=False,task_op=task_op)



        for epoch in range(1,self.args.n_epochs):
            self.pprint('Epoch:', epoch)
            self.pprint('training...')
       
            loss_sup = self.train_epoch_batched(self.model,optimizer,self.train_loader)

            self.pprint(loss_sup)

            self.pprint('evaluating...')

            
            train_score, train_loss = self.valid_batch(self.model, self.train_loader)

            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting training embeddings...')
                self.plot_embedding(self.embedding,self.labels,epoch,train_score,folder_path)

            valid_score, valid_loss = self.valid_batch(self.model, self.vali_loader)
            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting valid embeddings...')
                self.plot_embedding(self.valid_embedding,self.valid_labels,epoch,train_score,folder_path+'valid_plot/')

            test_score, test_loss = self.valid_batch(self.model, self.test_loader)
            if epoch % self.args.plot_epoch ==0:
                self.pprint('ploting test embeddings...')
                self.plot_embedding(self.valid_embedding,self.valid_labels,epoch,train_score,folder_path+'test_plot/')

            train_score_list.append(train_score)
            valid_score_list.append(valid_score)
            test_score_list.append(test_score)

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            test_loss_list.append(test_loss)

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
        self.plot_score(train_score_list,valid_score_list,test_score_list,folder_path,self.args)
        self.plot_loss(train_loss_list,valid_loss_list,test_loss_list,folder_path,self.args)
        self.plot_SSL_metrics(Log_ME_list,SFDA_list,NLEEP_list, folder_path,self.args)
        print("train data path: ",str(folder_path))

        return self.model
    
    def test(self, settings):
        output_path = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.args.log_file = os.path.join(output_path, 'run.log')
        
        if not self.args.is_training:
            self.test_loader = create_non_graph_loader(self.args,"test",shuffle=False,task_op=task_op)

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
            

            for batch_X, batch_y in tqdm(self.test_loader, total=len(self.test_loader)):
                output_y,_ = self.model.forward(batch_X.float().to(self.device))
                # print("y prob: ",output_y)
                # print("batch_y shape: ",batch_y.shape)
                y_prob = F.softmax(output_y, dim=1).cpu().numpy()
                # print("y prob: ",y_prob)
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)
                y_true = batch_y
                # print(y_pred,y_true)
                # print(torch.unique(batch_y))
                if self.args.dataset != "Epilepsy":
                    loss = criterion(output_y, y_true.squeeze().to(self.device))
                else:
                    loss = criterion(output_y, y_true.to(self.device))
            
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
        

    