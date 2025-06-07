import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
from constants import (
    ANOLE_PATH_HF,
    DATASET_TOKENIZED_PATH,
    TRANSFORMER_PATH,
    SAVE_DIR,
)
sys.path.append(TRANSFORMER_PATH)

import torch
from torch.nn import CrossEntropyLoss
import jsonlines
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import ChameleonForCausalLM, TrainingArguments
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEER_MATRIX_dir = SAVE_DIR
STEER_MATRIX_PATH = f"{STEER_MATRIX_dir}/epoch_1.pth"
MAX_VOCAB_SIZE = 65536
LEARNING_RATE=1e-6
NUM_TRAIN_EPOCHS=20
BATCH_SIZE=8
LOGGING_STEP=1

class SteerModel(ChameleonForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.STEER_matrix = self._initialize_steer_matrix()
        self._freeze_model_params()

    def _initialize_steer_matrix(self):
        lmhead_size = self.get_output_embeddings().weight.shape
        if os.path.exists(STEER_MATRIX_PATH):
            print(f"Loading STEER matrix from {STEER_MATRIX_PATH}")
            return nn.Parameter(torch.load(STEER_MATRIX_PATH), requires_grad=True).to(DEVICE)
        print("Initializing new STEER matrix.")
        matrix = torch.eye(lmhead_size[1], device=DEVICE)
        steer_param = nn.Parameter(matrix, requires_grad=True)
        # Save the parameter's underlying tensor (detached and moved to CPU if desired).
        torch.save(steer_param.detach().cpu(), STEER_MATRIX_PATH)
        return matrix

    def _freeze_model_params(self):
        for param in self.parameters():
            param.requires_grad = False
        self.STEER_matrix.requires_grad = True

    # Keep the original forward method
    @add_start_docstrings_to_model_forward(CHAMELEON_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import ChameleonProcessor, ChameleonForCausalLM
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> model = ChameleonForCausalLM.from_pretrained("meta-chameleon/meta-chameleon/chameleon-hf")
        >>> processor = ChameleonProcessor.from_pretrained("meta-chameleon/meta-chameleon/chameleon-hf")

        >>> image = Image.open(requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw)
        >>> image_2 = Image.open(requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw)
        >>> prompt = "What do these two images have in common?<image><image>"
        >>> inputs = processor(prompt, images=[image, image_2], return_tensors="pt").to(model.device, torch.float16)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        >>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(f"forward here:1{self.device}")
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        epsilon = 0.1
        emb = self.lm_head.weight

        logits = torch.matmul(hidden_states.to(DEVICE),((emb+ epsilon * torch.matmul(emb,self.STEER_matrix)).T))
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

class TokenizedDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                self.data.append((torch.tensor(obj['text_tokens'] + obj['image_tokens'], dtype=torch.long),
                                  torch.tensor(obj['label_tokens'], dtype=torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    batch_inputs = [torch.cat((item[0], item[1]), dim=0) for item in batch]
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=1)# <pad> is 1, check tokenmanager can prove
    batch_labels_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=-100)
    # Create attention masks
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != 1, 1)
   
    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': batch_labels_padded}



def train_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(),lr=LEARNING_RATE)
    model.train().to(DEVICE)

    for epoch in range(NUM_TRAIN_EPOCHS):
        loss_sum = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_TRAIN_EPOCHS}")):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if step % LOGGING_STEP == 0:
                print(f"Step {step} - Loss: {loss.item()}")
        print(f"Epoch {epoch + 1} Loss Avg: {loss_sum / len(dataloader)}")
        torch.save(model.STEER_matrix.data, f"{STEER_MATRIX_dir}/epoch_{epoch + 1}.pth")
    print("Training completed.")


if __name__ == "__main__":
    model = SteerModel.from_pretrained(ANOLE_PATH_HF)
    model.resize_token_embeddings(MAX_VOCAB_SIZE)

    dataset = TokenizedDataset(DATASET_TOKENIZED_PATH)


    train_model(model, dataset)
