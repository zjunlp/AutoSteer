import os
import sys
from pathlib import Path
FILE_PTH = sys.path.abspath(__file__)
STEER_PTH = Path(FILE_PTH).parent.parent.parent
sys.path.append(str(STEER_PTH/"steer"))

from steer.SteerLlava.steer_model_llava_inference import SteerModelForConditionalGeneration, Eval_Llava
from steer.SteerLlava.constants import model_path
from transformers import AutoProcessor, LlavaOnevisionConfig
from steer.SteerLlava.rater import LinearProber
import torch
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llava Embedding Generator with memmap")
    parser.add_argument('--input_file', type=str, required=True, help='Path to examine jsonl file')
    parser.add_argument('--save_dir', type=str, default="", help='Output dir')
    parser.add_argument('--img_dir', type=str, default="")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 8, 12, 16, 20, 24])


    args = parser.parse_args()
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # Processor and Model setup
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir=model_path)
    config = LlavaOnevisionConfig.from_pretrained(model_path + "/config")
    steer_config = {
        "gen": True,
        "DEVICE": args.device,
        "selected_layers": args.layers,
    }
    config.text_config.steer_config = steer_config

    model = SteerModelForConditionalGeneration.from_pretrained(
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        cache_dir=model_path,
        torch_dtype=torch.float16,
        config=config
    ).to(DEVICE)
    model.eval()

    eval = Eval_Llava(processor=processor)
    dataset = eval.get_unified_examine_data(pth=args.input_file, img_dir_pth=args.img_dir)
    print(f"Loaded {len(dataset)} samples")

    # Dry run on first sample to determine dim
    first_sample = dataset[0]
    model.set_first_pass()
    first_input = eval.process_llava_input(first_sample)
    first_input = first_input.to(DEVICE, torch.float16)
    try:
        _ = model.generate(**first_input, max_new_tokens=100, pad_token_id=151645)
    except StopIteration:
        pass
    embs = model.get_emb_h()
    hidden_dims = {layer: emb.shape[-1] for layer, emb in zip(args.layers, embs)}
    model.language_model.model.emb_h = []

    # Prepare open_memmap writers
    os.makedirs(args.save_dir, exist_ok=True)
    layer_files = {
        layer: np.lib.format.open_memmap(
            filename=os.path.join(args.save_dir, f"{layer}layer.npy"),
            mode='w+',
            dtype=np.float16,
            shape=(len(dataset), hidden_dims[layer])
        )
        for layer in args.layers
    }

    # Main loop
    with tqdm(dataset, desc="Processing dataset") as data_iter:
        for i, input in enumerate(data_iter):
            model.set_first_pass()
            input = eval.process_llava_input(input)
            input = input.to(DEVICE, torch.float16)

            try:
                _ = model.generate(**input, max_new_tokens=100, pad_token_id=151645)
            except(StopIteration):
                pass

            embs = model.language_model.model.emb_h

            for layer, emb in zip(args.layers, embs):
                layer_files[layer][i] = emb.cpu().numpy().astype(np.float16)


            model.language_model.model.emb_h = []

    print("✅ All embeddings saved with open_memmap.")