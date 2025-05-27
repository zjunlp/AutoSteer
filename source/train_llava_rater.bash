#NO VALIDATION
mkdir -p ./logs/train_rater/llava_rater_ckpt
for layer in {4..27..4}
do
  echo "Launching emb_layer=$layer"
  python train_rater.py --emb_layer $layer --num_epochs 200 --hidden_size 3584\
  --embs_dir "EMBS_PTH_LLAVA"\
  --save_ckpt_dir "RATER_CKPT_SAVE_DIR"\
  --device 0\
  --ratio 0\
  > ./logs/train_rater/llava_rater_ckpt/${layer}layer_llava.log 2>&1 &
done

wait
echo "All training jobs completed."

#VALIDATION SPLIT RATIO 0.1
mkdir -p ./logs/train_rater/llava_rater_ckpt_val
for layer in {4..27..4}
do
  echo "Launching emb_layer=$layer"
  python train_rater.py --emb_layer $layer --num_epochs 200 --hidden_size 3584\
  --embs_dir "EMBS_PTH_LLAVA"\
  --save_ckpt_dir "RATER_CKPT_SAVE_DIR"\
  --device 0\
  --ratio 0.1\
  > ./logs/train_rater/llava_rater_ckpt_val/${layer}layer_llava.log 2>&1 &
done

wait
echo "All training jobs completed."