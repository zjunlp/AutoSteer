# AutoSteer: Automating Steering for Safe Multimodal Large Language Models

## Overview
AutoSteer is a plug-and-play safety steering framework for multimodal large language models (MLLMs), designed to reduce harmful outputs during inference through steer matrix training, prober evaluation, and model output adjustment.

## 🚀 Get Started

### 🧩 Installation

#### 1. Download Models

**Chameleon**  
Download the model checkpoint and project from [GAIR-NLP/anole](https://github.com/GAIR-NLP/anole).

**Llava-OneVision**  
Use the following code snippet to download the model:

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig
import torch

cache_path = "your/model/path/llava-next-8b"
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir=cache_path)
config = LlavaOnevisionConfig.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir=cache_path)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    config=config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cuda:1",
    cache_dir=cache_path
)
```

---

#### 2. Download Image Datasets

- **COCO train2017**  
  Download and place in: `AutoSteer_final/dataset/COCO`

- **COCO2014**  
  Download and place in: `AutoSteer_final/dataset/COCO2014`

- **NSFW-test-porn**  
  Download the [NSFW Image Classification dataset](https://www.kaggle.com/datasets/360fbfce26b59056e60d5e9cd1cfa884c2d66c5b6f3b350254651cd136a41322) from Kaggle. Use the "porn" class from the test set. Replace the empty directory:
  ```
  AutoSteer_final/dataset/ToViLaG/porn
  ```

- **UCLA-protest**  
  Request from [here](https://github.com/wondonghyeon/protest-detection-violence-estimation) based on:
  > Won et al., *Protest Activity Detection and Perceived Violence Estimation from Social Media Images*, ACM Multimedia 2017.  
  Use the "protest" class from both train and test sets. Rename and place into:
  ```
  AutoSteer_final/dataset/ToViLaG/protest
  ```

- **Bloody Images**  
  Contact the [ToViLaG Author](mailto:wangxinpeng@tongji.edu.cn) to obtain bloody images. Replace:
  ```
  AutoSteer_final/dataset/ToViLaG/bloody
  ```

---

#### 3. Configure Constants

##### Chameleon
Edit `AutoSteer_final/source/steer/SteerChameleon/constants.py`:

```python
ckpt_path = "ANOLE/Anole-7b-v0.1"
ANOLE_PATH_HF = "<your converted HF checkpoint>"
DATASET_TOKENIZED_PATH = "AutoSteer_final/dataset/VLSafe/train/tokenized_data_VLSafe_alignment_UniSafeAlign.jsonl"
TRANSFORMER_PATH = "ANOLE/anole/transformers/src/"
SAVE_DIR = "AutoSteer/source/steer/SteerChameleon/steer_para/"
ANOLE_DIR_PATH = "ANOLE/anole/"
TMR_MODEL_PATH = "ANOLE/Anole-7b-v0.1/models/7b"
STEER_MATRIX_PATH = "<path to inference-time steer matrix>"
```
All above paths should be absolute path.

Edit `AutoSteer_final/source/LayerSelect/Chameleon/constants.py`:
```python
ANOLE_DIR_PATH = "ANOLE/anole/",
ckpt_path = "ANOLE/Anole-7b-v0.1"
```
All above paths should be absolute path.

##### Llava-OV
Edit `AutoSteer_final/source/steer/SteerLlava/constants.py`:

```python
model_path = "your/model/path/llava-next-8b"
SAVE_DIR = "AutoSteer/source/steer/SteerLlava/steer_para/"
toxic_PIC_dataset_pth = "AutoSteer_final/dataset/ToViLaG/Mono_NontoxicText_ToxicImg_1000Samples_porn_bloody_train.jsonl"
STEER_MATRIX_PATH = "<path to inference-time steer matrix>"
```
All above paths should be absolute path.

---

### 🔧 Run Training & Evaluation Pipelines

##### For **Chameleon**
```bash
cd AutoSteer_final/source
bash train_chameleon_steer.bash
bash select_layer_chameleon.bash
bash chameleon_prober_pipeline.bash
bash test_chameleon.bash
```

##### For **Llava-OneVision**
```bash
cd AutoSteer_final/source
bash train_llava_steer.bash
bash select_layer_llava.bash
bash llava_prober_pipeline.bash
bash test_llava.bash
```

---

## 📂 Directory Structure

```
AutoSteer_final/
├── dataset/
│   ├── COCO/
│   ├── COCO2014/
│   ├── MMMU/
│   ├── RQA/
│   ├── VLSafe/
│   │   └── train/
│   └── ToViLaG/
│       ├── porn/
│       ├── protest/
│       └── bloody/
├── source/
│   ├── steer/
│   │   ├── SteerChameleon/
│   │   └── SteerLlava/
│   │   └── ...
│   └── LayerSelect/
│   │   ├── Chameleon/
│   │   └── Llava/
│   │   └── ...
│   └── logs/
│   │   └── ...
├── model_output/
│   │   └── ...

```

---

## 📬 Contact

For any questions or issues, feel free to open an [issue](https://github.com/your-repo/issues) or contact the dataset authors as noted above.

---

## 📜 License

This project is released under the MIT License. See `LICENSE` for details.