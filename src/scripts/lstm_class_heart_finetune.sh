
python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_Heart_cluster_finetune_testratio1" \
    --seed 123 \
    --model "LSTM" \
    --dataset "Heartbeat" \
    --task "Detection" \
    --batch_size 32 \
    --max_clip_length 9 \
    --num_nodes 61 \
    --input_dim 45 \
    --output_dim 45 \
    --hidden_dim 8 \
    --num_rnn_layers 2 \
    --n_epochs 150 \
    --plot_epoch 50 \
    --early_stop 80 \
    --n_classes 1 \
    --plot_epoch 100 \
    --learning_rate 2e-3 \
    --weight_decay 1e-4 \
    --use_gpu True \
    --gpu "cuda:0" \
    --pretrain_model "LSTM_Pred" \
    --pretrain_model_path "LSTM_Heart_SSL_cluster_gs_final_LSTM_test" \
    --fine_tune \

