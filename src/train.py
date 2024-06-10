import argparse
import os
import torch
import random
import numpy as np
from exp.exp_classification import Exp_Classification
from exp.exp_detection import Exp_Detection
from exp.exp_ssl_eval import Exp_SSL_EVAL
from exp.exp_joint_eval import Exp_JOINT_EVAL
from exp.exp_joint_detection_eval import Exp_JOINT_DETECTION_EVAL
import cProfile


def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():


    parser = argparse.ArgumentParser(description='Hyperparameters Settings')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='LSTM',
                        help='model name, options: [LSTM]')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_batch', type=bool, default=True,help = "if use batched data")
    parser.add_argument('--dataset', type=str, default='Epilepsy', help='model name, options: [Epilepsy,HAR,Heartbeat]')    
    parser.add_argument('--task', type=str, default="Classification",choices = ["Classification","Detection","SSL","SSLDetection","SSLEval","SSLJoint","SSLJointDetection"],help = "task type")
    parser.add_argument('--cluster',action = 'store_true',help = 'use clustering loss')
    parser.add_argument('--contrastive',action = 'store_true',help = 'use contrastive loss')
    parser.add_argument('--supcon',action = 'store_true',help = 'use supervised contrastive loss')


    # data loader
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--max_clip_length', type=int, default=11, help='location of adj matrix')

    # model define
    parser.add_argument('--n_classes', type=int, default=4, help='class number')
    parser.add_argument('--seq_length', type=int, default=2500, help='input sequence length')
    parser.add_argument('--num_nodes', type=int, default=19, help='number of nodes')
    parser.add_argument('--input_dim', type=int, default=100, help='input channel')
    parser.add_argument('--output_dim', type=int, default=100, help='ouput channel')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden units')
    parser.add_argument('--cl_decay_steps', type=int, default=3000, help='Scheduled sampling decay steps.')
    parser.add_argument('--num_rnn_layers', type=int, default=3, help='input channel')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Maximum gradient norm for gradient clipping.')

    #pretrain model
    parser.add_argument('--fine_tune', action='store_true', help='use pretrained model')
    parser.add_argument('--pretrain_model', type=str, default="DCRNN_Pred", help='pretrain model')
    parser.add_argument('--pretrain_model_path', type=str, default="DCRNN_SSL1fft_DCRNN_test", help='pretrain model path')
    parser.add_argument('--pretrained_num_rnn_layers', type=int, default=3, help='pretrain model rnn layers')
    parser.add_argument('--pt_class', type=int, default=4, help='pretrain model fc class')
    parser.add_argument('--linear_probing', action='store_true', help='freeze the encoder')


    # others
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_activation', type=bool, default=True, help='use_activation')

    #auxiliary loss
    parser.add_argument('--cluster_attract_weight', type=float, default=0.25, help='weight for cos_loss')
    parser.add_argument('--cluster_repel_weight', type=float, default=0.25, help='weight for cos_loss')
    parser.add_argument('--cluster_prediction_weight', type=float, default=0.5, help='weight for cos_loss')
    parser.add_argument('--cluster_margin', type=float, default=0.25, help='margin of cluster')
    parser.add_argument('--w_main_task', type=float, default=0.5, help='w_main_task')
    parser.add_argument('--w_auxiliary_task', type=float, default=0.5, help='w_auxiliary_task')
    parser.add_argument('--drop_task_epoch', type=int, default=20, help='when drop auxiliary task')
    parser.add_argument('--aug_variance', type=float, default=0.5, help='variance of augmentation')


    # tSNE vis
    parser.add_argument('--plot_epoch', type=int, default=5, help='frequency of plotting embedding visualization')


    # optimization
    parser.add_argument('--N_WORKERS', type=int, default=0, help='data loader num workers')
    parser.add_argument('--prefetch_factor', type=int, default=None, help='pre-fetch')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--n_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--early_stop', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer regularization')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=str, default='cuda:2', help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')



    args = parser.parse_args()
    # args.use_gpu = False
    # print("use_gpu: ",args.use_gpu)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    fix_seed = args.seed
    seed_everything(fix_seed)

    print('Args in experiment:')
    print(args)


    if args.model == "GTS_CLASS_Dynamic":
        if args.task == "Detection":
            Exp = Exp_GTS_Dynamic_Detection
        else:
            Exp = Exp_GTS_Dynamic_Classification
            
    elif args.task == "Classification":
        Exp = Exp_Classification
    elif args.task == "SSLEval":
        Exp = Exp_SSL_EVAL
    elif args.task == "SSLJoint":
        Exp = Exp_JOINT_EVAL
    elif args.task == "SSLJointDetection":
        Exp = Exp_JOINT_DETECTION_EVAL
    else:
        Exp = Exp_Detection
    
    # Exp = Exp_GTS_Detection_Test




    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()

    else:         
        ii = 0
        setting = '{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.des, ii)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp = Exp(args)  # set experiments
        
        exp.test(setting)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    # cProfile.run("main()")