python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_EpilepsySmall_test" \
    --seed 123 \
    --model "LSTM" \
    --dataset "EpilepsySmall" \
    --batch_size 32 \
    --max_clip_length 2 \
    --num_nodes 3 \
    --input_dim 103 \
    --hidden_dim 16 \
    --num_rnn_layers 2 \
    --n_epochs 150 \
    --plot_epoch 100 \
    --early_stop 20 \
    --n_classes 4 \
    --learning_rate 3e-3 \
    --weight_decay 5e-4 \
    --use_gpu True \
    --gpu "cuda:1" \
