# export CUDA_VISIBLE_DEVICES=0


python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_HAR_SSL_contrastive_gs_final" \
    --model "LSTM" \
    --task "SSLEval" \
    --dataset "HAR" \
    --batch_size 128 \
    --max_clip_length 8 \
    --num_nodes 9 \
    --input_dim 16 \
    --output_dim 16 \
    --n_epochs 50 \
    --plot_epoch 10 \
    --early_stop 8 \
    --n_classes 6 \
    --learning_rate 5e-4 \
    --weight_decay 0e-4 \
    --use_gpu True \
    --gpu "cuda:0" \
    --contrastive \

