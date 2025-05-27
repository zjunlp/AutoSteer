mkdir -p ./logs/test/chameleon/mmmu
mkdir -p ./logs/test/chameleon/tovilag
mkdir -p ./logs/test/chameleon/vlsafe
mkdir -p ./logs/test/chameleon/rqa
#AUTO
python chameleon_steer_test.py --epsilon 0.1 --auto True \
    --test_dataset "../dataset/VLSafe/examine_sampled_500_VLSafe.jsonl" \
    --img_dir "../dataset/VLSafe/COCO/train2017/"\
    --output_file '../model_output/chameleon/vlsafe/eval_chameleon_autosteer_vlsafe_0dot1.jsonl' \
    --device 0\
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/vlsafe/eval_chameleon_autosteer_vlsafe_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0.1 --auto True \
    --test_dataset "../dataset/MMMU/MMMU_sample_500" \
    --output_file '../model_output/chameleon/mmmu/eval_chameleon_autosteer_mmmu_0dot1.jsonl' \
    --device 0 \
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/mmmu/eval_chameleon_autosteer_mmmu_0dot1.log" 2>&1

 python chameleon_steer_test.py --epsilon 0.1 --auto True  \
    --img_dir '../dataset/ToViLaG/' \
    --test_dataset "../dataset/ToViLaG/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test.jsonl" \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_autosteer_tovilag_toxic_img_porn_bloody_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_autosteer_tovilag_toxic_img_porn_bloody_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0.1 --auto True  \
    --img_dir '../dataset/COCO2014/COCO_2014/raw/COCO2014_test/' \
    --test_dataset '../dataset/ToViLaG/Mono_NontoxicImg_ToxicText_500Samples_convert4test.jsonl' \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_autosteer_tovilag_toxic_text_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_autosteer_tovilag_toxic_text_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0.1 --auto True  \
    --img_dir '../dataset/ToViLaG/' \
    --test_dataset "../dataset/ToViLaG/cotoxic_500Samples_convert4test2.jsonl" \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_autosteer_tovilag_cotoxic_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_autosteer_tovilag_cotoxic_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0.1 --auto True \
    --test_dataset "../dataset/RQA/realworld_qa_500_sample" \
    --output_file '../model_output/chameleon/rqa/eval_chameleon_autosteer_rqa_0dot1.jsonl' \
    --device 0 \
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/rqa/eval_chameleon_autosteer_rqa_0dot1.log" 2>&1

#STEER
python chameleon_steer_test.py --epsilon 0.1 --auto False \
    --test_dataset "../dataset/VLSafe/examine_sampled_500_VLSafe.jsonl" \
    --img_dir "../dataset/VLSafe/COCO/train2017/"\
    --output_file '../model_output/chameleon/vlsafe/eval_chameleon_steer_vlsafe_0dot1.jsonl' \
    --device 0\
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/vlsafe/eval_chameleon_steer_vlsafe_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0.1 --auto False \
    --test_dataset "../dataset/MMMU/MMMU_sample_500" \
    --output_file '../model_output/chameleon/mmmu/eval_chameleon_steer_mmmu_0dot1.jsonl' \
    --device 0 \
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/mmmu/eval_chameleon_steer_mmmu_0dot1.log" 2>&1

 python chameleon_steer_test.py --epsilon 0.1 --auto False  \
    --img_dir '../dataset/ToViLaG/' \
    --test_dataset "../dataset/ToViLaG/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test.jsonl" \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_steer_tovilag_toxic_img_porn_bloody_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_steer_tovilag_toxic_img_porn_bloody_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0.1 --auto False  \
    --img_dir '../dataset/COCO2014/COCO_2014/raw/COCO2014_test/' \
    --test_dataset '../dataset/ToViLaG/Mono_NontoxicImg_ToxicText_500Samples_convert4test.jsonl' \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_steer_tovilag_toxic_text_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_steer_tovilag_toxic_text_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0.1 --auto False  \
    --img_dir '../dataset/ToViLaG/' \
    --test_dataset "../dataset/ToViLaG/cotoxic_500Samples_convert4test2.jsonl" \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_steer_tovilag_cotoxic_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_steer_tovilag_cotoxic_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0.1 --auto False \
    --test_dataset "../dataset/RQA/realworld_qa_500_sample" \
    --output_file '../model_output/chameleon/rqa/eval_chameleon_steer_rqa_0dot1.jsonl' \
    --device 2 \
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/rqa/eval_chameleon_steer_rqa_0dot1.log" 2>&1
    
#ORIG
python chameleon_steer_test.py --epsilon 0 --auto False \
    --test_dataset "../dataset/VLSafe/examine_sampled_500_VLSafe.jsonl" \
    --img_dir "../dataset/VLSafe/COCO/train2017/"\
    --output_file '../model_output/chameleon/vlsafe/eval_chameleon_orig_vlsafe_0dot1.jsonl' \
    --device 0\
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/vlsafe/eval_chameleon_orig_vlsafe_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0 --auto False \
    --test_dataset "../dataset/MMMU/MMMU_sample_500" \
    --output_file '../model_output/chameleon/mmmu/eval_chameleon_orig_mmmu_0dot1.jsonl' \
    --device 0 \
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/mmmu/eval_chameleon_orig_mmmu_0dot1.log" 2>&1

 python chameleon_steer_test.py --epsilon 0 --auto False  \
    --img_dir '../dataset/ToViLaG/' \
    --test_dataset "../dataset/ToViLaG/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test.jsonl" \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_orig_tovilag_toxic_img_porn_bloody_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_orig_tovilag_toxic_img_porn_bloody_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0 --auto False  \
    --img_dir '../dataset/COCO2014/COCO_2014/raw/COCO2014_test/' \
    --test_dataset '../dataset/ToViLaG/Mono_NontoxicImg_ToxicText_500Samples_convert4test.jsonl' \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_orig_tovilag_toxic_text_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_orig_tovilag_toxic_text_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0 --auto False  \
    --img_dir '../dataset/ToViLaG/' \
    --test_dataset "../dataset/ToViLaG/cotoxic_500Samples_convert4test2.jsonl" \
    --output_file '../model_output/chameleon/tovilag/eval_chameleon_orig_tovilag_cotoxic_0dot1.jsonl' \
    --device 0 \
    --layer 24 \
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/tovilag/eval_chameleon_orig_tovilag_cotoxic_0dot1.log" 2>&1

python chameleon_steer_test.py --epsilon 0 --auto False \
    --test_dataset "../dataset/RQA/realworld_qa_500_sample" \
    --output_file '../model_output/chameleon/rqa/eval_chameleon_orig_rqa_0dot1.jsonl' \
    --device 0 \
    --layer 24\
    --rater_ckpt "./steer/chameleon_rater_ckpt_final/layer24/epoch_100.pt" \
    > "./logs/test/chameleon/rqa/eval_chameleon_orig_rqa_0dot1.log" 2>&1