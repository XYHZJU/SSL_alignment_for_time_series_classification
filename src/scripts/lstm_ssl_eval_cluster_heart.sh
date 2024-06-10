# export CUDA_VISIBLE_DEVICES=0


python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_Heart_SSL_cluster_gs_final" \
    --model "LSTM" \
    --task "SSLEval" \
    --dataset "Heartbeat" \
    --batch_size 128 \
    --max_clip_length 9 \
    --num_nodes 61 \
    --input_dim 45 \
    --output_dim 45 \
    --hidden_dim 8 \
    --num_rnn_layers 2 \
    --n_epochs 150 \
    --plot_epoch 20 \
    --early_stop 30 \
    --n_classes 2 \
    --cluster_attract_weight 0.00 \
    --cluster_repel_weight 0.00 \
    --cluster_prediction_weight 1.0 \
    --cluster_momentum 1 \
    --cluster_margin 1 \
    --learning_rate 9e-4 \
    --weight_decay 0e-4 \
    --use_gpu True \
    --gpu "cuda:0" \
    --cluster \
