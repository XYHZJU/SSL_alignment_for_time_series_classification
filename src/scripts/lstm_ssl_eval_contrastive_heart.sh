# export CUDA_VISIBLE_DEVICES=0


python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_Heart_SSL_supcontrastive_gs_final" \
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
    --n_epochs 200 \
    --plot_epoch 20 \
    --early_stop 80 \
    --n_classes 2 \
    --learning_rate 13e-4 \
    --weight_decay 0e-4 \
    --aug_variance 0.005 \
    --contrastive \
    --supcon \
    --use_gpu True \
    --gpu "cuda:0" \
