python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_HAR_joint" \
    --model "LSTM" \
    --task "SSLJoint" \
    --dataset "HAR" \
    --seed 123 \
    --batch_size 32 \
    --max_clip_length 8 \
    --drop_task_epoch 20 \
    --num_nodes 9 \
    --input_dim 16 \
    --output_dim 16 \
    --n_epochs 80 \
    --plot_epoch 80 \
    --early_stop 50 \
    --n_classes 6 \
    --cluster_attract_weight 0.0 \
    --cluster_repel_weight 0.0 \
    --cluster_prediction_weight 1 \
    --w_auxiliary_task 0.5 \
    --w_main_task 0.5 \
    --learning_rate 11e-4 \
    --weight_decay 0e-4 \
    --use_gpu True \
    --gpu "cuda:0" \


