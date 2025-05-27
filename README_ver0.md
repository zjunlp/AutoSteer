# AutoSteer: Automating Steering for Safe Multimodal Large Language Models
## Overview

## Get Started
### Installation
#### 1 Download Models
**Chameleon**
Refer https://github.com/GAIR-NLP/anole to download. Download model checkpoint and the anole project.

**Llava-OV**
Refer below python code to download Llava-OneVision to specified model path(cache_path).
```
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig
import torch
cache_path = "your/model/path/llava-next-8b"
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir=cache_path) 
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    config=config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cuda:1", 
    cache_dir=cache_path
)
```
#### 2 Download Image Datasets
##### **COCO train2017**

Download COCO 2017's train dataset, put the downloaded directory into AutoSteer_final/dataset/COCO

##### **COCO2014**

Download COCO2014 dataset, put the downloaded directory into AutoSteer_final/dataset/COCO2014.

##### **NSFW-test-porn**

Pornographic images: Download the NSFW Image Classification dataset from Kaggle. We use the porn class in the test set. Replace the empty porn dirctory(AutoSteer_final/dataset/ToViLaG/porn) with the downloaded NSDW-test-porn dataset dirctory.

##### **UCLA-protest**

Request UCLA Protest Image Dataset from [here](https://github.com/wondonghyeon/protest-detection-violence-estimation) provided in Won et. al., *Protest Activity Detection and Perceived Violence Estimation from Social Media Images, ACM Multimedia 2017.* We use the combination of the protest class from the train and test sets. Replace the empty protest directory(AutoSteer_final/dataset/ToViLaG/protest) with your downloaded and unzipped ucla-protest directory(rename it as protest).

##### **Bloody Images**

Please contact ToViLaG's Author via [email](https://github.com/victorup/ToViLaG/blob/main/wangxinpeng@tongji.edu.cn) to obtain the images. Replace the empty bloody dirctory(AutoSteer_final/dataset/ToViLaG/bloody).

#### 3 Run
We provide a through-out pipeline including steer matrix training, SAS layer selecting, prober training&evaluation and model evaluations. Model evaluations include original model, steered model and AutoSteer applied model evaluations. 
##### Settings Before Run
###### Chameleon

In AutoSteer_final/source/steer/SteerChameleon/constants.py, 
set ckpt_path to ANOLE/Anole-7b-v0.1,
    ANOLE_PATH_HF to your converted Anole hugging face checkpoint.(About how to convert, refer to [link](https://github.com/GAIR-NLP/anole)), 
    DATASET_TOKENIZED_PATH to your tokenized dataset(if you generated one yourself, following instructions from https://github.com/GAIR-NLP/anole) or the provided one AutoSteer_final/dataset/VLSafe/train/tokenized_data_VLSafe_alignment_UniSafeAlign.jsonl, 
    TRANSFORMER_PATH to your downloaded ANOLE/anole/transformers/src/ to import the package during training and inference,
    SAVE_DIR to store the trained checkpoint of steer matrixs(e.g.AutoSteer/source/steer/SteerChameleon/steer_para/),
    ANOLE_DIR_PATH to your downloaded ANOLE/anole/,
    TMR_MODEL_PATH to ANOLE/Anole-7b-v0.1/models/7b,
    STEER_MATRIX_PATH to where the steer matrix you want to use is located.

In AutoSteer_final/source/LayerSelect/Chameleon/constants.py,
set ANOLE_DIR_PATH to your downloaded ANOLE/anole/,
    ckpt_path to ANOLE/Anole-7b-v0.1,

###### Llava-OV
In AutoSteer_final/source/steer/SteerLlava/constants.py,
set model_path to above mentioned cache_path(your/model/path/llava-next-8b),
    SAVE_DIR to where the steer matrixs you want to store(e.g.AutoSteer/source/steer/SteerLlava/steer_para/),
    toxic_PIC_dataset_pth to AutoSteer_final/dataset/ToViLaG/,Mono_NontoxicText_ToxicImg_1000Samples_porn_bloody_train.jsonl.
    STEER_MATRIX_PATH to where the steer matrix you want to use during inference is located,

##### Chameleon

```
cd AutoSteer_final/source
bash train_chameleon_steer.bash
bash select_layer_chameleon.bash
bash chameleon_prober_pipeline.bash
bash test_chameleon.bash
```
##### Llava-OV
```
cd AutoSteer_final/source
bash train_llava_steer.bash
bash select_layer_llava.bash
bash llava_prober_pipeline.bash
bash test_llava.bash
```



