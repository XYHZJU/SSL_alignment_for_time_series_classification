# export CUDA_VISIBLE_DEVICES=0


python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_EpilepsySmall_SSL_autoPCA_test" \
    --model "LSTM" \
    --task "SSLEval" \
    --dataset "EpilepsySmall" \
    --batch_size 128 \
    --max_clip_length 2 \
    --num_nodes 3 \
    --input_dim 103 \
    --output_dim 103 \
    --hidden_dim 16 \
    --num_rnn_layers 2 \
    --n_epochs 100 \
    --plot_epoch 30 \
    --early_stop 10 \
    --n_classes 4 \
    --learning_rate 1e-3 \
    --weight_decay 0e-4 \
    --use_gpu True \
    --gpu "cuda:0" \
