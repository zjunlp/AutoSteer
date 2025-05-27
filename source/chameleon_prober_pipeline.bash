#CHAMELEON float 16 gen train&test
#gen train embs
mkdir -p ./logs/gen
python ./LayerSelect/Chameleon/gen_last_token_memmap_np.py \
    --input_file "../dataset/VLSafe/train/VLSafe_harmlessnss_alignment.jsonl" \
    --img_dir "../dataset/COCO/train2017/" \
    --save_dir "./LayerSelect/Chameleon/embs_for_prober_train/neg" \
    --device 1 \
    --layers $(seq 4 4 28) \
    > "./logs/gen/neg_Chameleon_train_prober.log" 2>&1

python ./LayerSelect/Chameleon/gen_last_token_memmap_np.py \
    --input_file "../dataset/VLSafe/train/vlsafe_convert_safe.jsonl" \
    --img_dir "../dataset/COCO/train2017/" \
    --save_dir "./LayerSelect/Chameleon/embs_for_prober_train/pos" \
    --device 1 \
    --layers $(seq 4 4 28) \
    > "./logs/gen/pos_Chameleon_train_prober.log" 2>&1
#gen test embs
mkdir -p ./logs/gen/neg
python ./LayerSelect/Chameleon/gen_last_token_memmap_np.py \
    --img_dir '../dataset/ToViLaG/' \
    --input_file '../dataset/ToViLaG/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test.jsonl' \
    --save_dir "./LayerSelect/Chameleon/embs_for_prober_test/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test" \
    --device 1 \
    --layers $(seq 4 4 28) \
    > "./logs/gen/neg/neg_Chameleon2.log" 2>&1

python ./LayerSelect/Chameleon/gen_last_token_memmap_np.py \
    --img_dir '../dataset/ToViLaG/' \
    --input_file '../dataset/ToViLaG/cotoxic_500Samples_convert4test2.jsonl' \
    --save_dir "./LayerSelect/Chameleon/embs_for_prober_test/cotoxic_500Samples_convert4test2" \
    --device 1 \
    --layers $(seq 4 4 28) \
    > "./logs/gen/neg/neg_Chameleon3.log" 2>&1

python ./LayerSelect/Chameleon/gen_last_token_memmap_np.py \
    --img_dir '../dataset/COCO2014/COCO_2014/raw/COCO2014_test/' \
    --input_file '../dataset/ToViLaG/Mono_NontoxicImg_ToxicText_500Samples_convert4test.jsonl' \
    --save_dir "./LayerSelect/Chameleon/embs_for_prober_test/Mono_NontoxicImg_ToxicText_500Samples_convert4test" \
    --device 1 \
    --layers $(seq 4 4 28) \
    > "./logs/gen/neg/neg_Chameleon1.log" 2>&1

python ./LayerSelect/Chameleon/gen_last_token_memmap_np.py \
    --input_file "../dataset/VLSafe/examine_sampled_500_VLSafe.jsonl" \
    --img_dir "../dataset/COCO/train2017/" \
    --save_dir "./LayerSelect/Chameleon/embs_for_prober_test/vlsafe" \
    --device 1 \
    --layers $(seq 4 4 28) \
    > "./logs/gen/neg/vlsafe_tmp_examine_Chameleon.log" 2>&1

mkdir -p ./logs/gen/pos
python ./LayerSelect/Chameleon/gen_last_token_memmap_np.py \
    --input_file "../dataset/RQA/realworld_qa_500_sample" \
    --save_dir "./LayerSelect/Chameleon/embs_for_prober_test/rqa" \
    --device 1 \
    --layers $(seq 4 4 28) \
    > "./logs/gen/pos/rqa_Chameleon.log" 2>&1
#Chameleon rater train
mkdir -p ./logs/train_rater/chameleon_rater_ckpt
for layer in {4..28..4}
do
  echo "Launching emb_layer=$layer"
  python train_rater.py --emb_layer $layer --num_epochs 200 --hidden_size 4096\
  --embs_dir "./LayerSelect/Chameleon/embs_for_prober_train"\
  --save_ckpt_dir "./steer/chameleon_rater_ckpt"\
  --device 1\
  --ratio 0\
  > ./logs/train_rater/chameleon_rater_ckpt/${layer}layer_Chameleon.log 2>&1 &
done

wait
echo "All training jobs completed."
#Chameleon rater eval
mkdir -p ./logs/evaluate_rater/chameleon
python evaluate_rater.py \
  --model_ckpt_dir ./steer/chameleon_rater_ckpt \
  --neg_embs_dir ./LayerSelect/Chameleon/embs_for_prober_test/vlsafe \
  --pos_embs_dir ./LayerSelect/Chameleon/embs_for_prober_test/rqa \
  --hidden_size 4096 \
  --layers 4 8 12 16 20 24 28 \
  --epochs $(seq 1 1 200)\
  --batch_size 1 \
  --device 1 \
  > "./logs/evaluate_rater/chameleon/Chameleon_rater_rqa_vlsafe.log" 2>&1
python evaluate_rater.py \
  --model_ckpt_dir ./steer/chameleon_rater_ckpt \
  --neg_embs_dir ./LayerSelect/Chameleon/embs_for_prober_test/cotoxic_500Samples_convert4test2 \
  --pos_embs_dir ./LayerSelect/Chameleon/embs_for_prober_test/rqa \
  --hidden_size 4096 \
  --layers 4 8 12 16 20 24 28 \
  --epochs $(seq 1 1 200)\
  --batch_size 1 \
  --device 1 \
  > "./logs/evaluate_rater/chameleon/Chameleon_rater_rqa_cotoxic.log" 2>&1
python evaluate_rater.py \
  --model_ckpt_dir ./steer/chameleon_rater_ckpt \
  --neg_embs_dir ./LayerSelect/Chameleon/embs_for_prober_test/Mono_NontoxicImg_ToxicText_500Samples_convert4test \
  --pos_embs_dir ./LayerSelect/Chameleon/embs_for_prober_test/rqa \
  --hidden_size 4096 \
  --layers 4 8 12 16 20 24 28 \
  --epochs $(seq 1 1 200)\
  --batch_size 1 \
  --device 1 \
  > "./logs/evaluate_rater/chameleon/Chameleon_rater_rqa_toxicText.log" 2>&1
python evaluate_rater.py \
  --model_ckpt_dir ./steer/chameleon_rater_ckpt \
  --neg_embs_dir ./LayerSelect/Chameleon/embs_for_prober_test/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test \
  --pos_embs_dir ./LayerSelect/Chameleon/embs_for_prober_test/rqa \
  --hidden_size 4096 \
  --layers 4 8 12 16 20 24 28 \
  --epochs $(seq 1 1 200)\
  --batch_size 1 \
  --device 1 \
  > "./logs/evaluate_rater/chameleon/Chameleon_rater_rqa_toxicImg.log" 2>&1