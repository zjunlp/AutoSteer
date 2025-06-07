import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
ckpt_path = Path(os.getenv("CKPT_PATH", "/mnt/16t/lyucheng/ANOLE/Anole-7b-v0.1"))
DATASET_RAW_PATH = Path("/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/VLSafe/train/output_alignment.jsonl")

# Tokenized dataset (specify the path that you want to store your tokenized dataset)
DATASET_TOKENIZED_PATH = Path("/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/VLSafe/train/tokenized_data_VLSafe_alignment_UniSafeAlign.jsonl")

# Tokenized dataset (specify the path that you want to store your images)
DATASET_IMAGE_PATH = Path("./images/")

# Anole torch path (specify the path that you want to store your Anole torch checkpoint)
ANOLE_PATH_TORCH =  ckpt_path / "models" / "7b"

TORCH_PATH = ckpt_path

# Anole HF path (specify the path that you want to store your Anole hugging face checkpoint)
ANOLE_PATH_HF = Path("/mnt/16t/lyucheng/ANOLE/Anole-7b-v0.1/model_HF")

# Anole HF path (specify the path that you want to store your fine-tuned Anole hugging face checkpoint)
ANOLE_PATH_HF_TRAINED = Path("/mnt/16t/lyucheng/ANOLE/training_CKPT/try2")

MODEL_7B_PATH = ckpt_path / "models" / "7b"

#------------------------------------------------------------------------------------#
TOKENIZER_TEXT_PATH = ckpt_path / "tokenizer" / "text_tokenizer.json"

TOKENIZER_IMAGE_PATH = ckpt_path / "tokenizer" / "vqgan.ckpt"

TOKENIZER_IMAGE_CFG_PATH = ckpt_path / "tokenizer" / "vqgan.yaml"
# Local package import path
ANOLE_DIR_PATH = "/mnt/16t/lyucheng/ANOLE/anole/" # abspath to ANOLE/anole dir, needed for chameleon import

TRANSFORMER_PATH = "/mnt/16t/lyucheng/ANOLE/anole/transformers/src/" # To import chameleon's transformer package

# COCO2017(for vlsafe)
COCO2017_PATH = "/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/COCO/train2017/"
# abspath to test datasets
MMMU_EXAMINE_PATH = "/mnt/20t/lyucheng/EvalDatasets/MMMU_sample_500"
RQA_EXAMINE_PATH = "/mnt/20t/lyucheng/EvalDatasets/realworld_qa_500_sample"
VLSAFE_EXAMINE_PATH = "/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/VLSafe/examine_sampled_500_VLSafe.jsonl"
