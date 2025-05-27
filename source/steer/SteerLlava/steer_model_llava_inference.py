import os
import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration,AutoModelForCausalLM, Qwen2ForCausalLM
from transformers import LlavaOnevisionConfig
from transformers import Qwen2Model
from typing import Callable, List, Optional, Tuple, Union
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM
from transformers.models.llava_onevision.modeling_llava_onevision import LlavaOnevisionCausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling
from .rater import LinearProber
from .constants import (model_path,
                        STEER_MATRIX_PATH,
                        RATER_CKPT_PATH,
                        )
import gc
import json
from datasets import load_from_disk
import re
import ast
from PIL import Image

logger = logging.get_logger(__name__)



class ProbeModel(Qwen2Model):
    def __init__(self, config,gen = False, selected_layer = 24, selected_layers = [*range(4,29,4)],need_rating = False, fix_steer_epsilon = 0, rater_ckpt_pth = RATER_CKPT_PATH, threshhold = 0.5, safety_ratio=1.0):
        super().__init__(config)
        steer_config = config.steer_config
        print(steer_config)
        DEVICE = f"cuda:{steer_config.get('DEVICE')}" if torch.cuda.is_available() else "cpu"

        self.fix_steer_epsilon = steer_config.get("epsilon") if steer_config.get("epsilon") is not None else 0
        self.auto_epsilon = None
        self.need_rating = steer_config.get("need_rating", False)
        self.get_toxicity = steer_config.get("get_toxicity", False)
        #print(self.fix_steer_epsilon)
        if self.fix_steer_epsilon == 0:
            print("Use unsteer model!")
        else:
            if self.need_rating:
                print(f"AutoSteer applied, base epsilon is {self.fix_steer_epsilon}")
            else:
                print(f"Steer is applied, fixed epsilon is {self.fix_steer_epsilon}")
        rater_ckpt_pth = steer_config.get("rater_ckpt_pth", rater_ckpt_pth)
        if self.need_rating:
            if rater_ckpt_pth is not None:
                self.rater = LinearProber(DEVICE=DEVICE,checkpoint_pth=rater_ckpt_pth,hidden_sizes=[3584,64,2]).to(DEVICE).to(dtype=torch.float16)
            else:
                self.rater = LinearProber(DEVICE=DEVICE).to(DEVICE)

        self.first_pass = False  # Flag of first pass, signaling for auto-epsilon
        self.DEVICE = DEVICE

        self.threshhold = steer_config.get("threshhold", 0.5)
        self.safety_ratio = steer_config.get("safety_ratio", 1.0)
        self.selected_layer = steer_config.get("selected_layer", 20)  # For prober to specify layer

        # Generate emb related
        self.selected_layers = steer_config.get("selected_layers", list(range(4, 28, 4)))  # For emb generation
        self.gen = steer_config.get("gen", False)
        self.emb_h = []
    
    def set_first_pass(self,first_pass = True):
        self.first_pass = first_pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i,decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            #print(f"This is layer [{i}]")
            if self.gen and self.first_pass and (i in self.selected_layers): 
                # print(f"selected layer is {i}")
                #print(f"hidden states' shape:{hidden_states.shape}")# 1,seq_len,emb_size
                #print(f"len of emb_h is {len(self.emb_h)}")
                self.emb_h.append(hidden_states[0][-1]) # take last token emb
                # rate = self.rate(hidden_states[0][-1])
                # print(hidden_states[0][-1])
                # print(hidden_states[0][-1].dtype)
                # print(f"rate is {rate}")
                # if rate > self.threshhold:
                #     print(f"toxicity exceeds threshold: {rate}")

            if self.need_rating and (i == self.selected_layer) and self.first_pass:
                # emb = hidden_states[0]
                # print(f"selected rating layer is {i}")
                # print(emb_16th.shape)
                rate = self.rate(hidden_states[0][-1])
                if self.get_toxicity:
                    self.toxicity = rate
                if rate > self.threshhold:
                    print(f"toxicity exceeds threshold: {rate}")
                    epsilon = self.safety_ratio * self.fix_steer_epsilon # 0.05 is empirically good
                else : epsilon = 0
                self.auto_epsilon = epsilon

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        self.first_pass = False
        
        if self.gen:
            raise StopIteration("Forwarding not required during embedding generation.")
        

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def rate(self, emb):
        #emb: seq_len*emb_size
        # print(f"emb shape {emb.shape}")
        outputs = self.rater(emb)
        # print(f"rater output is :{outputs}")
        toxicity = outputs[0] #idx 0 for toxicity possibility, within range [0, 1]
        # print(f"toxicity is :{toxicity}")
        return toxicity

class SteerModel(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        steer_config = config.steer_config
        self.model = ProbeModel(config)
        DEVICE = f"cuda:{steer_config.get('DEVICE')}" if torch.cuda.is_available() else "cpu"
        self.STEER_matrix = self._initialize_steer_matrix(DEVICE= DEVICE)
        self._freeze_model_params()
        
        self.fix_steer_epsilon = steer_config.get('epsilon') if steer_config.get('epsilon') is not None else 0


    def _initialize_steer_matrix(self,DEVICE):
        lmhead_size = self.get_output_embeddings().weight.shape
        if os.path.exists(STEER_MATRIX_PATH):
            print(f"Loading STEER matrix from {STEER_MATRIX_PATH}")
            matrix = torch.load(STEER_MATRIX_PATH).to(DEVICE)
            steer_param = nn.Parameter(matrix.float())  # Don't call .to() after nn.Parameter
        else:
            print("Initializing new STEER matrix.")
            matrix = torch.eye(lmhead_size[1], device=DEVICE)
            steer_param = nn.Parameter(matrix.float())
            torch.save(steer_param.detach().cpu(), STEER_MATRIX_PATH)
        return steer_param

    def _freeze_model_params(self):
        for param in self.parameters():
            param.requires_grad = False
        self.STEER_matrix.requires_grad = True
    
    def get_auto_epsilon(self):
        if self.model.need_rating:
            print(f"Auto epsilon is {self.model.auto_epsilon}")
            return self.model.auto_epsilon
        else:
            return self.fix_steer_epsilon
    
    # def set_fix_steer_epsilon(self,epsilon =0):
    #     self.fix_steer_epsilon = epsilon
    #     self.model.fix_steer_epsilon = epsilon

    # def set_gen(self,gen = False):
    #     self.model.gen = gen

    # def set_selected_layer(self,selected_layer = 24):
    #     self.model.selected_layer = selected_layer

    # def set_selected_layers(self,selected_layers = [*range(4,29,4)]):
    #     self.model.selected_layers = selected_layers  

    # def set_need_rating(self,need_rating = False):
    #     self.model.need_rating = need_rating


    # def set_threshhold(self,threshhold = 0.5):
    #     self.model.threshhold = threshhold

    # def set_safety_ratio(self,safety_ratio=1.0):
    #     self.model.safety_ratio = safety_ratio
    
    def get_generated_emb(self):
        return self.model.emb_h

    def set_first_pass(self,first_pass = True):
        self.model.first_pass = first_pass
    
    def set_emb_h(self,emb_h = []):
        self.model.emb_h = emb_h
        #print(f"len of emb_h is {len(self.model.emb_h)} in set_emb_h")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        # logits = self.lm_head(hidden_states[:, slice_indices, :])

        epsilon = self.get_auto_epsilon()
        #emb = self.lm_head.weight
        emb = self.lm_head.weight.float()       # 将语言模型头的权重转换为 float16
        #print(f"steer epsilon is {epsilon}")
        logits = torch.matmul(hidden_states[:, slice_indices, :].float(),((emb+ epsilon * torch.matmul(emb,self.STEER_matrix)).T))

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class SteerModelForConditionalGeneration(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(self, "language_model"):
            del self.language_model
            gc.collect()
            torch.cuda.empty_cache()
        self.language_model = SteerModel._from_config(config.text_config)

        # self.epsilon = 0
        # self.gen = False
        # self.selected_layer = 24
        # self.selected_layers = [*range(4,29,4)]
        # self.need_rating = False
        # self.DEVICE = "cuda"
        # self.rater_ckpt_pth = None
        # self.threshhold = 0.5
        # self.safety_ratio=1.0

    def set_epsilon(self,epsilon =0):
        self.language_model.set_fix_steer_epsilon(epsilon)

    def set_gen(self,gen = False):
        self.language_model.set_gen(gen)

    def set_probe_layer(self,selected_layer = 24):
        self.language_model.set_selected_layer(selected_layer)

    def set_selected_layers(self,selected_layers = [*range(4,29,4)]):
        self.language_model.set_selected_layers(selected_layers)

    def set_need_rating(self,need_rating = False):
        self.language_model.set_need_rating(need_rating)

    def set_threshhold(self,threshhold = 0.5):
        self.language_model.set_threshhold(threshhold)


    def set_safety_ratio(self,safety_ratio=1.0):
        self.language_model.set_safety_ratio(safety_ratio)

    def set_parameters(self, epsilon=0, gen=False, probe_layer=24, selected_layers=[*range(4, 29, 4)], 
                        need_rating=False, threshhold=0.5, safety_ratio=1.0):
        self.language_model.set_fix_steer_epsilon(epsilon)
        self.language_model.set_gen(gen)
        self.language_model.set_selected_layer(probe_layer)
        self.language_model.set_selected_layers(selected_layers)
        self.language_model.set_need_rating(need_rating)
        self.language_model.set_threshhold(threshhold)
        self.language_model.set_safety_ratio(safety_ratio)

    def get_emb_h(self):
        return self.language_model.get_generated_emb()
    
    def set_emb_h(self,emb_h = []):
        self.language_model.set_emb_h(emb_h)
        
    def set_first_pass(self,first_pass = True):
        self.language_model.set_first_pass(first_pass)

class Eval_Llava:
    def __init__(self,processor,max_input_len = 131072):
        self.processor = processor
        self.max_length = max_input_len

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
                        {"type": "text", "content": data.get("query") if data.get("query") is not None else data.get("safe_query")},
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
                    {"type": "text", "content": query},# query
                    {"type": "image", "content": image},
                    {"type": "answer", "content": ans}
                ])
        else:
            raise ValueError("Unrecognized dataset format. The input data does not match any known structure.")
        
        return input_to_chameleon
    
    def get_mmmu_examine_data(self,pth ):
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
    


    def get_vlsafe_examine_data(self, pth="/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/VLSafe/examine_sampled_500_VLSafe.jsonl",img_dir_pth = "/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/COCO/train2017/"):
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
    def get_vlasfe_convert_safe_data(pth="/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/VLSafe/vlsafe_convert_safe.jsonl",img_dir_pth = "/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/COCO/train2017/"):
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
                    {"type": "text", "content": data["safe_query"]},
                    {"type": "image", "content": img_dir_pth + data["image_id"]}
                ])
                
        return my_input

    def get_vlsafe_alignment_data(pth="/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/VLSafe/VLSafe_harmlessnss_alignment.jsonl",img_dir_pth = "/mnt/16t/lyucheng/ANOLE/dataset/train_dataset/COCO/train2017/"):
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
    
    def process_llava_input(self,input_data):
        question = None
        image = None
        for input_seg in input_data:
            if input_seg["type"] == "text":
                #print(f"\nquery: {input_seg['content']}\n")
                question = input_seg["content"]
            elif input_seg["type"] == "image":
                if isinstance(input_seg["content"], str):
                    abs_path = os.path.abspath(input_seg["content"])
                    if os.path.exists(abs_path):
                        image = Image.open(abs_path)
                        image = image.resize((256,256))
                    else:
                        print(f"警告: 图片路径 {abs_path} 不存在")
                elif isinstance(input_seg["content"], Image.Image):
                    image = input_seg["content"].resize((256,256))
                    #print("Image provided dirctly.")
                else:
                    print("not path or Image")
            elif input_seg["type"] == "question_id":
                # 存储问题ID以供后续使用
                self.question_id = input_seg["content"]
            elif input_seg["type"] == "answer":
                #print(f"answer is {input_seg['content']}")
                pass
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]},
        ]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, add_image_tokens=True)# Must be consistent with the prompt template used in training, prober training included
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )
        return inputs
