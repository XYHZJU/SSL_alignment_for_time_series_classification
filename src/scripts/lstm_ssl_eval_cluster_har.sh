# export CUDA_VISIBLE_DEVICES=0

python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_HAR_SSL_cluster_final2" \
    --model "LSTM" \
    --task "SSLEval" \
    --dataset "HAR" \
    --batch_size 32 \
    --max_clip_length 8 \
    --num_nodes 9 \
    --input_dim 16 \
    --output_dim 16 \
    --n_epochs 50 \
    --plot_epoch 50 \
    --early_stop 10 \
    --n_classes 6 \
    --learning_rate 5e-4 \
    --weight_decay 0e-4 \
    --cluster_attract_weight 0.25 \
    --cluster_repel_weight 0.25 \
    --cluster_prediction_weight 0.5 \
    --cluster_momentum 1 \
    --use_gpu True \
    --cluster \
    --gpu "cuda:0" \


