# Self-Supervised Learning Task Alignment for Time Series Classification

Measure alignment between the self-supervised pre-training task and downstream tasks on multi variable time series data.

## Project Overview

Self-Supervised Learning (SSL) is getting popular for improving the performance of Neural Networks on supervised ML tasks on time series. Many auxiliary tasks are designed to extract the temporal, spatial or semantic information of time series and learn a better latent representation, and SSL as a kind of auxiliary learning method is widely used as they don't need labels for optimization . There are mainly two ways to train auxiliary tasks: pre-training & fine-tuning and joint-learning. The motivation of the pre-training & fine-tuning strategy is that the weights learned by the encoder on the auxiliary task provide a warm start for the model on the downstream task and therefore improve its performance. The joint-learning aims to train the auxiliary tasks together with the downstream task and it is believed that the combination of tasks can in the end lead to a lower loss for the downstream task.
Nonetheless, in some cases even though the model achieved good performance on a SSL task, the downstream performance does not improve, or even get worse, so the SSL tasks and the downstream tasks are not always aligned.

To address the problem, the thesis proposes a supervised clustering method that could enhance the alignment between the autoregressive SSL task and downstream classification tasks. The thesis also quantifies the alignment with the transferability metrics LogME, SFDA and NLEEP, and visualized embeddings of different auxiliary tasks through PCA and t-SNE.  Two learning paradigms are compared: the joint-learning paradigm and the pre-training and fine-tuning paradigm. The experiments were done on three multivariate time series classification datasets using LSTM as the backbone, and the results show that autoregressive SSL with supervised clustering aligns most with the downstream tasks in the aspect of improving the accuracy, and the pre-training strategy works better than joint-learning for such task.

Finally, the thesis discusses task alignment based on the motivation of representation learning, and derived the conclusion that supervised clustering is a good regularization for the autoregressive SSL task to improve the downstream task performance.


## Start
The needed packages are in the `requirements.yaml` file.
Then, you should create a virtual environment by running
```sh
make env
```
The environment can be activated with
```sh
conda activate ./env
```

The figures in our experiments are stored in the `figure` folder. 

The logs and checkpoints are in the `checkpoints` folder.

The embedding plots and training curves generated during training are in the `train_loss` folder.

The bash scripts for launching the experiment are in the `src/scripts` folder.

## Datasets

The datasets used for the projects are HAR, Epilepsy and Heartbeat.

In order to get the preprocessed HAR dataset, run the code:

```sh
python src/data/HAR_preprocess.py
```

In order to get the preprocessed Epilepsy or Heartbeat dataset, first specify the name of the dataset in UEA_preprocess.py by changing the last line to 

```sh
load_data('Epilepsy') / load_data('Heartbeat')
```
and then generate the corresponding dataset by running the script:

```sh
python src/data/UEA_preprocess.py
```

## Hyperparameters

There are some explanations about the hyperparameters in the experiments:

`--task` when performing downstream classification task for HAR and Epilepsy, the value is `Classification`, when performing detection task for Heartbeat, the value is `Detection`. When performing pretraining, the value is `SSLEval`, when performing joint-classification, the value is `SSLJoint`, when performing joint-detection, the value is `SSLJointDetection`.

`--max_clip_length` is the value of number of clips for each sample.

`--num_nodes` is the number of dimensions for multivariate time series.

`--input_dim` is the value of length for each clip in the sample.

`--pretrain_model` and `--pretrain_model_path` are the name and path of pre-trained model for finetuning.

`--cluster_attract_weight` and `--cluster_repel_weight` is the weights of supervised clustering.

`--cluster_prediction_weight` is the weight of autoregressive SSL.

`--cluster_margin` is the margin for inter class distance loss.

`--w_main_task` and `--w_auxiliary_task` are the weights for joint-learning.

`--drop_task_epoch` is the epoch number for dropping the auxiliary task in joint-learning.

`--aug_variance` is the variance value for augmentation in contrastive learning.

`--plot_epoch` is the interval for generate all kinds of plottings during training.

Besides, there are also some `store_true` args which controls the state of training:

`--cluster` is whether using supervised clustering during pre-training.

`--contrastive` is whether use self-supervised contrastive learning, and if this is set true with `--supcon`, then it is supervised contrastive learning.

`--fine_tune` is whether finetuning the pre-trained model.

`--linear_probing` is whether freezing the encoder during fine-tuning.







--------

<p><small>Project based on the <a target="_blank" href="https://github.com/LTS4/cookiecutter-lts4-student-project/">cookiecutter LTS4 student project template</a>. </small></p>
