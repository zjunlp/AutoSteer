python ./steer/SteerLlava/steer_model_llava_train.py \
    --batch_size 2 \
    --num_epochs 30 \
    --lr 1e-5 \
    --dataset_path "../dataset/train_dataset/VLSafe/train/output_alignment.jsonl" \
    --dataset_PIC_path "../dataset/ToViLaG/Mono_NontoxicText_ToxicImg_1000Samples_porn_bloody_train.jsonl" \
    --output_dir "../source/steer/SteerLlava/steer_para" \
    --pretrained_model "llava-hf/llava-onevision-qwen2-7b-ov-hf"\
    >  "../source/logs/steer_train/llava_steer_train.log" 2>&1 


