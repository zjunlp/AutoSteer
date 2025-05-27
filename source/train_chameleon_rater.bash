#NO VALIDATION
for layer in {4..28..4}
do
  echo "Launching emb_layer=$layer"
  python train_rater.py --emb_layer $layer --num_epochs 200 --hidden_size 4096\
  --embs_dir "EMBS_PTH_LLAVA"\
  --save_ckpt_dir "RATER_CKPT_SAVE_DIR"\
  --device 0\
  --ratio 0\
  > ./logs/train_rater/chameleon_rater_ckpt/${layer}layer_chameleon.log 2>&1 &
done

wait
echo "All training jobs completed."

#VALIDATION SPLIT RATIO 0.1
for layer in {4..28..4}
do
  echo "Launching emb_layer=$layer"
  python train_rater.py --emb_layer $layer --num_epochs 200 --hidden_size 4096\
  --embs_dir "EMBS_PTH_LLAVA"\
  --save_ckpt_dir "RATER_CKPT_SAVE_DIR"\
  --device 0\
  --ratio 0.1\
  > ./logs/train_rater/chameleon_rater_ckpt/${layer}layer_chameleon_0dot1_val.log 2>&1 &
done

wait
echo "All training jobs completed."