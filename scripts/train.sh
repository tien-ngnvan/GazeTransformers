SAVED_PATH='output/pretrain-detr'

rm -rf $SAVED_PATH

CUDA_VISIBLE_DEVICES=0,1 \

python -m src.train \
    --model_name_or_path 'facebook/detr-resnet-50' \
    --output_dir $SAVED_PATH \
    --overwrite_output_dir \
    --do_train \
    --dataset_name "tiennv/gaze-following-short" \
    --object_path 'scripts/objects.txt' \
    --num_train_epochs 50 \
    --per_device_train_batch_size 4 \
    --learning_rate 3e-5 \
    --weight_decay 1e-2 \
    --warmup_ratio 0.1 \
    --save_steps 5000 \
    --save_total_limit 3 \
    --logging_steps 500 \
    --fp16 \
    --remove_unused_columns False \
    --debug 'underflow_overflow'