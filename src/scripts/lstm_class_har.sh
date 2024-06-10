python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_HAR_456" \
    --seed 456 \
    --model "LSTM" \
    --dataset "HAR" \
    --batch_size 32 \
    --max_clip_length 8 \
    --num_nodes 9 \
    --input_dim 16 \
    --n_epochs 50 \
    --plot_epoch 10 \
    --early_stop 8 \
    --n_classes 6 \
    --learning_rate 11e-4 \
    --weight_decay 3e-4 \
    --use_gpu True \
    --gpu "cuda:1" \