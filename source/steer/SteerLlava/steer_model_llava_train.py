import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from typing import List, Optional, Tuple, Union

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2ForCausalLM
from transformers.models.llava_onevision.modeling_llava_onevision import LlavaOnevisionCausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from constants import (model_path,
                        toxic_PIC_dataset_pth,
                        SAVE_DIR
                        )
STEER_MATRIX_PATH = SAVE_DIR + "/epoch_1.pth"
import jsonlines
import json
from tqdm import tqdm
from PIL import Image
import gc
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SteerModel(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.STEER_matrix = self._initialize_steer_matrix()
        self._freeze_model_params()

    def _initialize_steer_matrix(self):
        lmhead_size = self.get_output_embeddings().weight.shape
        if os.path.exists(STEER_MATRIX_PATH):
            print(f"Loading STEER matrix from {STEER_MATRIX_PATH}")
            matrix = torch.load(STEER_MATRIX_PATH).to(DEVICE)
            steer_param = nn.Parameter(matrix.float())
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

        epsilon = 0.1
        emb = self.lm_head.weight.float()       

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


    

def steer_collate_fn(batch):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=151643)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    
    pixel_values = torch.stack([item["pixel_values"] for item in batch])  # images should be resized to same size
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    image_sizes = torch.stack([item["image_sizes"] for item in batch])    

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_sizes": image_sizes,
    }

class SteerDataset(Dataset):
    def __init__(self, json_path, processor, max_length=131072):
        self.data = []
        with jsonlines.open(json_path) as reader:
            for obj in reader:
                self.data.append(obj)
        self.data = self.data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"])
        image = image.resize((256,256))  # Resize image to a fixed size
        question = item["text"]
        answer = "I'm sorry, but I can't assist with that."  # Refusal answer

        conversation = [
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, add_image_tokens=True)
        #print(f"Prompt text: {prompt_text}")
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        image_token_index = (inputs["input_ids"] == image_token_id).nonzero(as_tuple=True)[1][-1].item()
        labels[:, :image_token_index + 1] = -100  # 包括 <image> 位置之前的全部设为忽略

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "image_sizes": torch.tensor(image.size[::-1]), #
        }
class SteerDataset_PIC(Dataset):
    def __init__(self, json_path, processor, max_length=131072):
        self.data = []
        with jsonlines.open(json_path) as reader:
            for obj in reader:
                self.data.append(obj)
        self.data = self.data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(toxic_PIC_dataset_pth+item["image_pth"])
        image = image.resize((256,256))  # Resize image to a fixed size
        question = item["query"]
        answer = "I'm sorry, but I can't assist with that."  # Placeholder answer

        # 构造多模态对话模板
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, add_image_tokens=True)
        #print(f"Prompt text: {prompt_text}")
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()
        # Mask掉输入部分，只监督生成回答
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        image_token_index = (inputs["input_ids"] == image_token_id).nonzero(as_tuple=True)[1][-1].item()
        labels[:, :image_token_index + 1] = -100  # 包括 <image> 位置之前的全部设为忽略

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "image_sizes": torch.tensor(image.size[::-1]), #
        }
def train(model, dataloader, optimizer, num_epochs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            assert labels.max() < 151936#model.config.vocab_size
            image_sizes = batch["image_sizes"].to(DEVICE)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.language_model.STEER_matrix.detach().cpu(), f"{output_dir}/Epoch_{epoch+1}.pt")

    print("TRAIN DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SteerModel with specified parameters.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for the optimizer.")
    parser.add_argument("--dataset_path", type=str,required=True, help="Path to the training dataset JSONL file.")
    parser.add_argument("--dataset_PIC_path", type=str,required=True, help="Path to the training dataset JSONL file.")
    parser.add_argument("--cache_dir", type=str, default=model_path, help="Cache directory for model files.")
    parser.add_argument("--output_dir", type=str, default="./steer_para", help="Directory to save STEER matrix checkpoints.")
    parser.add_argument("--pretrained_model", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", help="Pretrained model name or path.")

    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.pretrained_model, cache_dir=args.cache_dir)
    model = SteerModelForConditionalGeneration.from_pretrained(args.pretrained_model, cache_dir=args.cache_dir, torch_dtype=torch.float16).to(DEVICE)
    #model = model.to(torch.float32)    
    model.train()

    dataset = SteerDataset(args.dataset_path, processor)
    if args.dataset_PIC_path is not None:
        dataset2 = SteerDataset_PIC(args.dataset_PIC_path, processor)
        dataset = ConcatDataset([dataset, dataset2])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=steer_collate_fn)##

    optimizer = torch.optim.AdamW([model.language_model.STEER_matrix], lr=args.lr)

    num_epochs = args.num_epochs
    train(model, dataloader, optimizer, num_epochs,output_dir=args.output_dir)



