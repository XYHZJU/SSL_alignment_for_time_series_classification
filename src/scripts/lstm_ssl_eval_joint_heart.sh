python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_Heart_joint" \
    --model "LSTM" \
    --task "SSLJointDetection" \
    --dataset "Heartbeat" \
    --seed 123 \
    --batch_size 32 \
    --max_clip_length 9 \
    --drop_task_epoch 80 \
    --num_nodes 61 \
    --input_dim 45 \
    --output_dim 45 \
    --n_epochs 50 \
    --plot_epoch 50 \
    --early_stop 50 \
    --n_classes 1 \
    --cluster_attract_weight 0 \
    --cluster_repel_weight 0 \
    --cluster_prediction_weight 1 \
    --contrastive \
    --supcon \
    --cluster_margin 1 \
    --learning_rate 2e-3 \
    --weight_decay 1e-4 \
    --use_gpu True \
    --gpu "cuda:0" \


