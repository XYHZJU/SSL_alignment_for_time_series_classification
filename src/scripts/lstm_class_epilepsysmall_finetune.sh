python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_EpilepsySmall_cluster_finetune_testratio0.8" \
    --seed 123 \
    --model "LSTM" \
    --dataset "EpilepsySmall" \
    --batch_size 128 \
    --max_clip_length 2 \
    --num_nodes 3 \
    --input_dim 103 \
    --output_dim 103 \
    --hidden_dim 16 \
    --num_rnn_layers 2 \
    --n_epochs 150 \
    --early_stop 40 \
    --n_classes 4 \
    --plot_epoch 80 \
    --learning_rate 3e-3 \
    --weight_decay 5e-4 \
    --use_gpu True \
    --gpu "cuda:0" \
    --pretrain_model "LSTM_Pred" \
    --pretrain_model_path "LSTM_EpilepsySmall_SSL_cluster_testratio_LSTM_test" \
    --fine_tune \