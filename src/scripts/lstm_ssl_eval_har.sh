# export CUDA_VISIBLE_DEVICES=0


python -u src/train.py \
    --is_training 1 \
    --model_id "LSTM_HAR_SSL_gs_autoPCAtest" \
    --model "LSTM" \
    --task "SSLEval" \
    --dataset "HAR" \
    --batch_size 32 \
    --max_clip_length 8 \
    --num_nodes 9 \
    --input_dim 16 \
    --output_dim 16 \
    --n_epochs 50 \
    --plot_epoch 5 \
    --early_stop 8 \
    --n_classes 6 \
    --clip_length 11 \
    --clip_stride 1 \
    --pre_ictal_length 1 \
    --high_seizure 1000 \
    --learning_rate 11e-4 \
    --weight_decay 0e-4 \
    --use_gpu True \
    --gpu "cuda:0" \
    --use_fft