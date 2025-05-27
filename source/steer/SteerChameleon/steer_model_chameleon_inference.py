import os
from os.path import abspath,dirname
import torch
import sys
from .constants import (
    ANOLE_DIR_PATH,
    TRANSFORMER_PATH,
    MODEL_7B_PATH,
    STEER_MATRIX_PATH
)
# cur_dir_path = abspath(dirname(__file__))
# sys.path.append(os.path.join(dirname(dirname(dirname(dirname(cur_dir_path)))),'ANOLE/anole'))
sys.path.append(ANOLE_DIR_PATH)
from torch.nn import CrossEntropyLoss
import jsonlines
from chameleon.inference import loader
from torchviz import make_dot
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import ChameleonForCausalLM
from torch import nn
# sys.path.append('/mnt/16t/lyucheng/ANOLE/anole/training/')
# sys.path.append('/mnt/16t/lyucheng/ANOLE/anole/chameleon/')
from chameleon.inference.transformer import Transformer

import pandas as pd
tmr_pth = MODEL_7B_PATH
from torch.optim import AdamW
import json
from tqdm import tqdm
# os.environ["TORCH_USE_CUDA_DSA"]=str(1)
#os.environ["CUDA_LAUNCH_BLOCKING"]=str(1)
from transformers.optimization import Adafactor
from PIL.Image import Image
from datasets import load_from_disk
sys.path.append(TRANSFORMER_PATH)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.chameleon.modeling_chameleon import (
    CHAMELEON_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from .rater import LinearProber
import re
import ast

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_VOCAB_SIZE = 65536
LEARNING_RATE=1e-6
NUM_TRAIN_EPOCHS=10
BATCH_SIZE=8
LOGGING_STEP=1

# Stimulate ChameleonInferenceModel
class ModelArgs:
    model_parallel_size: int = 1
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 65536
    ffn_dim_multiplier: float | None = 1.0
    multiple_of: int = 256
    norm_eps: float = 1e-05
    rope_theta: float = 10000.0
    qk_normalization: bool = True ###Orig True
    swin_norm: bool = False

class SteerModel(Transformer):
    def __init__(self, args =ModelArgs(),get_toxicity = False, gen = False, selected_layer = 24, selected_layers = [*range(4,29,4)],need_rating = False, epsilon = 0, DEVICE = None, rater_ckpt_pth = None, threshhold = 0.5, safety_ratio=1.0, STEER_MATRIX_PATH = STEER_MATRIX_PATH
):
        super().__init__(args)
        self.STEER_matrix = self._initialize_steer_matrix(STEER_MATRIX_PATH,DEVICE)       
        self._freeze_model_params()
        self.epsilon = epsilon
        self.need_rating = need_rating
        self.get_toxicity = get_toxicity
        if epsilon == 0:
            print("Use unsteer model!")
        else :
            if need_rating:
                print(f"AutoSteer applied, base epsilon is {epsilon}")
            else:
                print(f"Steer is applied, fixed epsilon is {epsilon}")
        if self.need_rating:
            if rater_ckpt_pth is not None:
                self.rater = LinearProber(DEVICE=DEVICE,checkpoint_pth=rater_ckpt_pth).to(DEVICE)
            else:
                self.rater = LinearProber(DEVICE=DEVICE).to(DEVICE)
        self.first_pass = False # Flag of first pass, signaling for auto-epsilon
        
        self.DEVICE = DEVICE
        self.threshhold = threshhold
        self.safety_ratio = safety_ratio
        self.selected_layer = selected_layer # For prober to specify layer
        #generate emb related
        self.selected_layers = selected_layers # For emb generation
        self.gen = gen
        self.emb_h = []


    def _initialize_steer_matrix(self,STEER_MATRIX_PATH,DEVICE):
        emb_size = 4096
        if os.path.exists(STEER_MATRIX_PATH):
            print(f"Loading STEER matrix from {STEER_MATRIX_PATH}")
            return nn.Parameter(torch.load(STEER_MATRIX_PATH,map_location=DEVICE).to(torch.bfloat16), requires_grad=True)
        print("Initializing new STEER matrix.")
        matrix = nn.Parameter(torch.eye(emb_size).to(DEVICE), requires_grad=True)
        torch.save(matrix, STEER_MATRIX_PATH)
        return matrix

    def _freeze_model_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def set_epsilon(self,epsilon =0):
        self.epsilon = epsilon

    @torch.no_grad()
    def forward_with_attn_bias(
        self,
        token_values: torch.Tensor,
        attn_bias,
        cache,
        group = None,
    ) -> torch.Tensor:
        # print(token_values)
        # print(type(attn_bias))
        # print(attn_bias)
        # print(type(cache))
        # print(cache)
        # h = self.tok_embeddings(token_values) ####.to(torch.float)
        h = self.tok_embeddings(token_values) ####.to(torch.float)

        for i, layer in enumerate(self.layers):
            #print(f"h {i} shape: {h.shape}")
            #print(f"h {i}: {h}")
            if self.gen and self.first_pass and (i in self.selected_layers): 
                self.emb_h.append(h[-1]) # take last token emb
                
            
            if self.need_rating and i == self.selected_layer and self.first_pass:
                emb = h
                # print(emb_16th.shape)
                rate = self.rate(emb[-1])
                if self.get_toxicity:
                    self.toxicity = rate
                if rate > self.threshhold:
                    print(f"toxicity exceeds threshold: {rate}")
                    auto_epsilon = self.safety_ratio * self.epsilon # 0.05 is empirically good
                else : auto_epsilon = 0
                self.auto_epsilon = auto_epsilon

            h = layer(h, cache[i], attn_bias, group=group)
        
        self.first_pass = False
        if self.gen:
            raise StopIteration("Forwarding not required during embedding generation.")
        
        emb = self.output.weight
        epsilon = self.auto_epsilon if self.need_rating else self.epsilon
        print(f"steer epsilon is {epsilon}")
        if epsilon != 0:
            logits = torch.matmul(self.norm(h).to(self.DEVICE),((emb+ epsilon * torch.matmul(emb,self.STEER_matrix)).T))
        else:
            logits = torch.matmul(self.norm(h).to(self.DEVICE),emb.T)
        # logits = self.tmr.output(self.tmr.norm(h))

        return logits.float()
    
    def rate(self, emb):
        #emb: seq_len*emb_size
        #print(f"emb shape {emb.shape}")
        outputs = self.rater(emb)
        # print(f"rater output is :{outputs}")
        toxicity = outputs[0] #idx 0 for toxicity possibility, within range [0, 1]
        print(f"toxicity is :{toxicity}")
        return toxicity
    # def auto_rate(self,tok_emb,vis_emb):
    #     epsilon = self.balancer.calculate(tok_emb,vis_emb)[0] # returns (epsilon, final_score)
    #     epsilon /= 10
    #     return epsilon
    
    # def set_epsilon(self,epsilon):
    #     self.balancer.set_autosteer(auto=False)
    #     self.balancer.set_epsilon(epsilon=epsilon)
    
    # def set_need_rating(self,need=True):
    #     self.need_rating=need

    # def set_autosteer(self,auto=True):
    #     self.balancer.set_autosteer(auto=auto)

    def generate_emb(self,token_values):
        print(token_values)
        h = self.tok_embeddings(token_values)
        return h


class Eval_Chameleon:
    def get_unified_examine_data(self, pth, img_dir_pth="/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/COCO/train2017/"):
        # Processes input data from various formats and returns a unified structure 
        # compatible with the chameleon interface.
        # Args:
        #     pth (str): Path to the dataset. If the path ends with `.jsonl`, the 
        #         VLSafe format is used. Otherwise, the dataset is loaded using 
        #         `load_from_disk` and processed based on its structure.
        #     img_dir_pth (str, optional): Root directory path for images when using 
        #         the VLSafe format. Defaults to 
        #         "/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/COCO/train2017/".
        # Returns:
        #     list: A list of dictionaries, where each dictionary represents a 
        #     processed input item compatible with the chameleon interface.
        # Raises:
        #     ValueError: If the dataset is empty or the format is unrecognized.
        # Supported Formats:
        #     1. VLSafe and ToviLag Format:
        #         - Input is a `.jsonl` file.
        #         - Each line contains a JSON object with `query` and `image_pth` fields.
        #         - Returns a list of dictionaries with "text" and "image" types.
        #     2. MMMU Format:
        #         - Dataset contains `image_1`, `options`, and `question` fields.
        #         - Options are converted to a prompt with lettered choices.
        #         - Returns a list of dictionaries with "text", "image", and "answer" types.
        #     3. realworld_qa Format:
        #         - Dataset contains `image`, `question`, and `answer` fields.
        #         - Returns a list of dictionaries with "text", "image", and "answer" types.
        # Notes:
        #     - Images are resized to 256x256 pixels for MMMU and realworld_qa formats.
        #     - For MMMU format, options stored as strings are parsed into lists.
        #     - If the dataset format does not match any of the supported structures, 
        #       an error is raised.
        # """
        # 根据传入的路径，统一返回不同格式数据的标准接口。
        
        # 参数：
        #     pth: 数据路径。当路径以 .jsonl 结尾时采用 VLSafe 格式；
        #         否则使用 load_from_disk 加载，并根据数据字段区分 MMMU 或 realworld_qa 格式。
        #     img_dir_pth: 如果使用 VLSafe 格式，则需要传入图片所在的根目录路径。
        
        # 返回：
        #     处理后的输入列表，其内各项为符合 chameleon 接口要求的字典列表。
        # """
        # VLSafe 格式：直接从 JSONL 文件中读取
        if pth.endswith(".jsonl"):
            my_input = []
            with open(pth, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    my_input.append([
                        {"type": "text", "content": data.get("query")if data.get("query") is not None else data.get("safe_query")},
                        {"type": "image", "content": os.path.join(img_dir_pth, data.get("image_id") if data.get("image_id") is not None else data.get("image_pth"))}
                    ])
            return my_input

        # 非 JSONL 格式，通过 load_from_disk 加载数据
        dataset = load_from_disk(pth)
        input_to_chameleon = []
        if not dataset:
            raise ValueError("Dataset is empty. Please check the path or the dataset format.")

        # 判断数据格式，参考第一个样本
        sample = dataset[0]
        # MMMU 格式：包含 "image_1" 和 "options"
        if 'image_1' in sample and 'options' in sample:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for item in dataset:
                image = item['image_1'].resize((256, 256))
                query = item['question']
                options = item['options']
                # 如果选项以字符串形式存储，则转为列表
                if isinstance(options, str):
                    options = ast.literal_eval(options)

                # 构建带选项字母标识的 prompt 文本
                prompt_text = query.strip() + "\n"
                for i, option in enumerate(options):
                    prompt_text += f"{letters[i]}. {option}\n"
                prompt_text += "Please answer directly with only the letter of the correct option."

                # 可选：对 prompt_text 按 <image 数字> 分割（此处保留原逻辑）
                parts = re.split(r'<image\s*\d+>', prompt_text)
                part1 = parts[0].strip()
                part2 = parts[1].strip() if len(parts) > 1 else ''

                ans = item['answer']
                input_to_chameleon.append([
                    {"type": "text", "content": part1},
                    {"type": "image", "content": image},
                    {"type": "text", "content": part2},
                    {"type": "answer", "content": ans}
                ])
        # realworld_qa 格式：包含 "image" 和 "answer"
        elif 'image' in sample and 'answer' in sample:
            for item in dataset:
                image = item['image'].resize((256, 256))
                query = item['question']
                ans = item['answer']
                input_to_chameleon.append([
                    {"type": "text", "content": query},
                    {"type": "image", "content": image},
                    {"type": "answer", "content": ans}
                ])
        else:
            raise ValueError("Unrecognized dataset format. The input data does not match any known structure.")
        
        return input_to_chameleon
    def get_mmmu_examine_data(self,pth):
        dataset = load_from_disk(pth)
        input_to_chameleon = []
        for item in dataset:
            image = item['image_1'].resize((256, 256))
            query = item['question']
            options = item['options']
            
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if isinstance(options,str):
                options = ast.literal_eval(options)
            #print(len(options))
            prompt_text = query.strip() + "\n"
            for i in range(len(options)):
                prompt_text += letters[i]+". "+options[i]+"\n"

            prompt_text += "Please answer directly with only the letter of the correct option."

            parts = re.split(r'<image\s*\d+>', prompt_text)
            part1 = parts[0].strip()
            part2 = parts[1].strip() if len(parts) > 1 else ''

            ans = item['answer']
            input_to_chameleon.append([
                    {"type": "text", "content": part1},
                    {"type": "image", "content": image},
                    {"type": "text", "content": part2},
                    {"type": "answer", "content": ans}
                ])
        return input_to_chameleon
    
    def get_realworld_qa_examine_data(self,pth = "/mnt/20t/lyucheng/EvalDatasets/realworld_qa_500_sample"):
        dataset = load_from_disk(pth)
        input_to_chameleon = []
        for item in dataset:
            image = item['image'].resize((256, 256))
            query = item['question']
            ans = item['answer']
            input_to_chameleon.append([
                    {"type": "text", "content": query},
                    {"type": "image", "content": image},
                    {"type": "answer", "content": ans}
                ])
        return input_to_chameleon

    def get_mllmguard_noise_examine_data(self,pth = "/mnt/16t/lyucheng/ANOLE/dataset/MLLMGuard/desensitize/noise-injection/en.csv"):
        df = pd.read_csv(pth)
        # 显示前几行数据
        print(df.head())

        # 显示数据基本信息
        print(df.info())

        # 显示统计信息
        print(df.describe())

    def get_vlasfe_examine_data(self, pth="/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/VLSafe/examine_sampled_500_VLSafe.jsonl",img_dir_pth = "/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/COCO/train2017/"):
        """获取VLSafe评估数据
        
        Args:
            pth: VLSafe数据集路径
            
        Returns:
            list: 包含评估数据的列表
        """
        
        my_input = []
        
        # 读取VLSafe数据
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                my_input.append([
                    {"type": "text", "content": data["query"]},
                    {"type": "image", "content": img_dir_pth + data["image_id"]}
                ])
                
        return my_input
    
    def get_examine_data(self, pth="/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/VLSafe/examine_sampled_500_VLSafe.jsonl",img_dir_pth = "/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/COCO/train2017/"):
        """可以用于获取ToviLaG Plus和vlsafe评估数据,目的是作一个通用的接口
        
        Args:
            pth: ToviLaG Plus数据集路径
            
        Returns:
            list: 包含评估数据的列表
        """
        
        my_input = []
        
        # 读取VLSafe数据
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                my_input.append([
                    {"type": "text", "content": data["query"]},
                    {"type": "image", "content": img_dir_pth + data["image_pth"]}
                ])
                
        return my_input
    
    def split_token_sequence(
        self,
        tokens: torch.LongTensor,
        boi: int,
        eoi: int
    ) -> List[Tuple[str, torch.LongTensor]]:
        batch_size, _ = tokens.shape
        assert batch_size == 1, "Batch size must be 1"
        
        device = tokens.device
        tokens = tokens[0]
        tokens = tokens.to(device)
        segments = []
        current_segment = []
        in_image_seg = False

        for token in tokens:
            if token == boi:
                if current_segment:
                    segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                    current_segment = []
                in_image_seg = True
            elif token == eoi and in_image_seg:
                segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
                in_image_seg = False
            else:
                current_segment.append(token)
        if current_segment:
            if in_image_seg:
                segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            else:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        return segments

    def process_chameleon_input(self, input_data):
        """处理输入数据,转换为Chameleon模型需要的格式
        
        Args:
            input_data: 输入数据列表,每个元素包含type和content字段
            
        Returns:
            batch_prompt_ui: 处理后的输入格式 
        """
        batch_prompt_ui = [[]]
        for input_seg in input_data:
            if input_seg["type"] == "text":
                #print(f"\nquery: {input_seg['content']}\n")
                batch_prompt_ui[0].append(
                    {"type": "text", "value": input_seg["content"]}
                )
            elif input_seg["type"] == "image":
                if isinstance(input_seg["content"], str):
                    abs_path = os.path.abspath(input_seg["content"])
                    if os.path.exists(abs_path):
                        batch_prompt_ui[0].append(
                            {"type": "image", "value": f"file:{abs_path}"}
                        )
                    else:
                        print(f"警告: 图片路径 {abs_path} 不存在")
                elif isinstance(input_seg["content"], Image):
                    batch_prompt_ui[0].append(
                        {"type": "image", "value": input_seg["content"]}
                    )
                    print("Image provided dirctly.")
                else:
                    print("not path or Image")
            elif input_seg["type"] == "question_id":
                # 存储问题ID以供后续使用
                self.question_id = input_seg["content"]
            elif input_seg["type"] == "answer":
                #print(f"answer is {input_seg['content']}")
                pass
        return batch_prompt_ui

    def process_chameleon_output(self, model,output_data, img_save_dir,gen_number,boi=8197, eoi=8196):
        segments = self.split_token_sequence(output_data, boi, eoi)
        
        os.makedirs(img_save_dir, exist_ok=True)
        output = ""
        for seg_id, (seg_type, seg_tokens) in enumerate(segments):
            if seg_type == "image_seg":
                assert seg_tokens.shape[1] == 1024
                img: Image = model.decode_image(seg_tokens)[0]
                image_path = os.path.join(img_save_dir, f"{str(gen_number)}_{seg_id}.png")
                img.save(image_path)
                output += f"<img: {image_path}>"
            else:
                assert seg_type == "text_seg"
                decoded_text = model.decode_text(seg_tokens)[0]
                output += decoded_text
        return output
    
    def process_chameleon_output_(self, token_manager,output_data, img_save_dir,gen_number,boi=8197, eoi=8196):
        # print(output_data)
        segments = self.split_token_sequence(output_data, boi, eoi)
        # print(segments)
        if img_save_dir is not None:
            os.makedirs(img_save_dir, exist_ok=True)
            print("image dir provided.")
        output = ""
        for seg_id, (seg_type, seg_tokens) in enumerate(segments):
            if seg_type == "image_seg":
                print("gen a image.")
                assert seg_tokens.shape[1] == 1024
                img: Image = token_manager.decode_image(seg_tokens)[0]
                if img_save_dir is not None:
                    image_path = os.path.join(img_save_dir, f"{str(gen_number)}_{seg_id}.png")
                    img.save(image_path)
                    output += f"<img: {image_path}>"
                else: 
                    output += f"<image: {seg_id}>"
            else:
                assert seg_type == "text_seg"
                decoded_text = token_manager.decode_text(seg_tokens)[0]
                output += decoded_text
        return output
