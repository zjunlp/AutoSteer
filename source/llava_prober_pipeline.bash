#LLAVA float 16 gen train&test
##gen train embs
mkdir -p ./logs/gen
python ./LayerSelect/Llava/gen_last_token_memmap_np.py \
    --input_file "../dataset/VLSafe/train/VLSafe_harmlessnss_alignment.jsonl" \
    --img_dir "../dataset/COCO/train2017/" \
    --save_dir "./LayerSelect/Llava/embs_for_prober_trainl/neg" \
    --device 1 \
    --layers $(seq 4 4 24) \
    > "./logs/gen/neg_llava_train_prober.log" 2>&1

python ./LayerSelect/Llava/gen_last_token_memmap_np.py \
    --input_file "../dataset/VLSafe/train/vlsafe_convert_safe.jsonl" \
    --img_dir "../dataset/COCO/train2017/" \
    --save_dir "./LayerSelect/Llava/embs_for_prober_trainl/pos" \
    --device 1 \
    --layers $(seq 4 4 24) \
    > "./logs/gen/pos_llava_train_prober.log" 2>&1
##gen test embs
mkdir -p ./logs/gen/neg
python ./LayerSelect/Llava/gen_last_token_memmap_np.py \
    --img_dir '../dataset/ToViLaG/' \
    --input_file '../dataset/ToViLaG/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test.jsonl' \
    --save_dir "./LayerSelect/Llava/embs_for_prober_test/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test" \
    --device 1 \
    --layers $(seq 4 4 24) \
    > "./logs/gen/neg/neg_llava2.log" 2>&1

python ./LayerSelect/Llava/gen_last_token_memmap_np.py \
    --img_dir '../dataset/ToViLaG/' \
    --input_file '../dataset/ToViLaG/cotoxic_500Samples_convert4test2.jsonl' \
    --save_dir "./LayerSelect/Llava/embs_for_prober_test/cotoxic_500Samples_convert4test2" \
    --device 1 \
    --layers $(seq 4 4 24) \
    > "./logs/gen/neg/neg_llava3.log" 2>&1

python ./LayerSelect/Llava/gen_last_token_memmap_np.py \
    --img_dir '../dataset/COCO2014/COCO_2014/raw/COCO2014_test/' \
    --input_file '../dataset/ToViLaG/Mono_NontoxicImg_ToxicText_500Samples_convert4test.jsonl' \
    --save_dir "./LayerSelect/Llava/embs_for_prober_test/Mono_NontoxicImg_ToxicText_500Samples_convert4test" \
    --device 1 \
    --layers $(seq 4 4 24) \
    > "./logs/gen/neg/neg_llava1.log" 2>&1

python ./LayerSelect/Llava/gen_last_token_memmap_np.py \
    --input_file "../dataset/VLSafe/train/examine_sampled_500_VLSafe.jsonl" \
    --img_dir "../dataset/COCO/train2017/" \
    --save_dir "./LayerSelect/Llava/embs_for_prober_test/vlsafe" \
    --device 1 \
    --layers $(seq 4 4 24) \
    > "./logs/gen/neg/vlsafe_examine_llava.log" 2>&1

mkdir -p ./logs/gen/pos
python ./LayerSelect/Llava/gen_last_token_memmap_np.py \
    --input_file "../dataset/RQA/realworld_qa_500_sample" \
    --save_dir "./LayerSelect/Llava/embs_for_prober_test/rqa" \
    --device 1 \
    --layers $(seq 4 4 24) \
    > "./logs/gen/pos/rqa_llava.log" 2>&1
#LLAVA rater train
mkdir -p ./logs/train_rater/llava_rater_ckpt
for layer in {4..27..4}
do
  echo "Launching emb_layer=$layer"
  python train_rater.py --emb_layer $layer --num_epochs 200 --hidden_size 3584\
  --embs_dir "./LayerSelect/Llava/embs_for_prober_trainl"\
  --save_ckpt_dir "./steer/llava_rater_ckpt"\
  --device 2\
  --ratio 0\
  > ./logs/train_rater/llava_rater_ckpt/${layer}layer_llava_padSet.log 2>&1 &
done

wait
echo "All training jobs completed."
#LLAVA rater eval
mkdir -p ./logs/evaluate_rater/llava
python evaluate_rater.py \
  --model_ckpt_dir ./steer/llava_rater_ckpt \
  --neg_embs_dir ./LayerSelect/Llava/embs_for_prober_test/vlsafe \
  --pos_embs_dir ./LayerSelect/Llava/embs_for_prober_test/rqa \
  --hidden_size 3584 \
  --layers 4 8 12 16 20 24 \
  --epochs $(seq 1 1 200)\
  --batch_size 1 \
  --device 1 \
  > "./logs/evaluate_rater/llava/llava_rater_llava_rater_ckptrqa_vlsafe.log" 2>&1
python evaluate_rater.py \
  --model_ckpt_dir ./steer/llava_rater_ckpt \
  --neg_embs_dir ./LayerSelect/Llava/embs_for_prober_test/cotoxic_500Samples_convert4test2 \
  --pos_embs_dir ./LayerSelect/Llava/embs_for_prober_test/rqa \
  --hidden_size 3584 \
  --layers 4 8 12 16 20 24 \
  --epochs $(seq 1 1 200)\
  --batch_size 1 \
  --device 1 \
  > "./logs/evaluate_rater/llava/llava_rater_llava_rater_ckptrqa_cotoxic.log" 2>&1
python evaluate_rater.py \
  --model_ckpt_dir ./steer/llava_rater_ckpt \
  --neg_embs_dir ./LayerSelect/Llava/embs_for_prober_test/Mono_NontoxicImg_ToxicText_500Samples_convert4test \
  --pos_embs_dir ./LayerSelect/Llava/embs_for_prober_test/rqa \
  --hidden_size 3584 \
  --layers 4 8 12 16 20 24 \
  --epochs $(seq 1 1 200)\
  --batch_size 1 \
  --device 1 \
  > "./logs/evaluate_rater/llava/llava_rater_llava_rater_ckptrqa_toxicText.log" 2>&1
python evaluate_rater.py \
  --model_ckpt_dir ./steer/llava_rater_ckpt \
  --neg_embs_dir ./LayerSelect/Llava/embs_for_prober_test/Mono_NontoxicText_ToxicImg_500Samples_porn_bloody_convert4test \
  --pos_embs_dir ./LayerSelect/Llava/embs_for_prober_test/rqa \
  --hidden_size 3584 \
  --layers 4 8 12 16 20 24 \
  --epochs $(seq 1 1 200)\
  --batch_size 1 \
  --device 1 \
  > "./logs/evaluate_rater/llava/llava_rater_llava_rater_ckptrqa_toxicImg.log" 2>&1
