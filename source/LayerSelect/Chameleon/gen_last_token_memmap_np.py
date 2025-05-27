import os
import sys
import argparse
import torch
import inspect
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List
from PIL import Image
import json

FILE_PTH = sys.path.abspath(__file__)
STEER_PTH = Path(FILE_PTH).parent.parent.parent
sys.path.append(str(STEER_PTH/"steer"))

from constants import (
    ANOLE_DIR_PATH,
    TOKENIZER_TEXT_PATH,
    TOKENIZER_IMAGE_CFG_PATH,
    TOKENIZER_IMAGE_PATH,
    MODEL_7B_PATH
)
sys.path.append(ANOLE_DIR_PATH)

from chameleon.inference import loader
from chameleon.inference.transformer import ModelArgs
from chameleon.inference.chameleon import ChameleonInferenceModel, Options, TokenManager, Generator
from SteerChameleon.steer_model_chameleon_inference import SteerModel, Eval_Chameleon

def load_steer_model(path: str,
                     device: str,
                     selected_layers: List[int],
                     epsilon: float = 0,
                     auto: bool = False,
                     rater_ckpt_pth: str = None,
                     safety_ratio: float = 1,
                     rank: int = 0,
                     gen: bool = False) -> SteerModel:
    src_dir = Path(path)
    with open(src_dir / "params.json", "r") as f:
        params = json.load(f)
    with open(src_dir / "consolidate_params.json", "r") as f:
        consolidate_params = json.load(f)
    params = {**params, **params["model"], **consolidate_params}
    known_param = inspect.signature(ModelArgs.__init__).parameters
    filtered_params = {k: v for k, v in params.items() if k in known_param}
    model_args = ModelArgs(**filtered_params)
    model = SteerModel(
        args=model_args,
        epsilon=epsilon,
        need_rating=auto,
        rater_ckpt_pth=rater_ckpt_pth,
        safety_ratio=safety_ratio,
        DEVICE=device,
        selected_layers=selected_layers,
        gen=gen
    )
    ckpt_path = src_dir / "consolidated.pth"
    if not ckpt_path.exists():
        ckpt_path = src_dir / f"consolidated.{rank:02}.pth"
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).to(torch.float16)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Chameleon Embedding Generator with memmap")
    parser.add_argument('--input_file', type=str, required=True, help='Path to examine jsonl')
    parser.add_argument('--save_dir', type=str, required=True, help='Output .npy memmap dir')
    parser.add_argument('--img_dir', type=str, default="")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 8, 12, 16, 20, 24,28])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(DEVICE)
    torch.cuda.set_device(args.device)

    # Load model and token manager
    eval = Eval_Chameleon()
    dataset = eval.get_unified_examine_data(pth=args.input_file, img_dir_pth=args.img_dir)
    print(f"Loaded dataset: {len(dataset)} samples")

    token_manager = TokenManager(
        tokenizer_path=TOKENIZER_TEXT_PATH.as_posix(),
        vqgan_cfg_path=TOKENIZER_IMAGE_CFG_PATH.as_posix(),
        vqgan_ckpt_path=TOKENIZER_IMAGE_PATH.as_posix(),
        device=DEVICE,
    )

    model_kwargs = {
        "path": MODEL_7B_PATH,
        "safety_ratio": 1,
        "device": DEVICE,
        "selected_layers": args.layers,
        "gen": True
    }
    Smodel = load_steer_model(**model_kwargs)
    options = Options()

    # Dry run to get hidden dim
    sample_input = eval.process_chameleon_input(dataset[0])
    batch_input_ids = [token_manager.tokens_from_ui(ui) for ui in sample_input]
    Smodel.first_pass = True
    try:
        for _ in Generator(Smodel, token_manager.vocab, options, batch_input_ids):
            pass
    except StopIteration:
        pass
    embs = Smodel.emb_h
    hidden_dims = {layer: emb.shape[-1] for layer, emb in zip(args.layers, embs)}
    Smodel.emb_h = []

    # Open memmap writers
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
    for i, input in enumerate(tqdm(dataset, desc="Generating embeddings")):
        input = eval.process_chameleon_input(input)
        batch_input_ids = [token_manager.tokens_from_ui(ui) for ui in input]
        Smodel.first_pass = True
        try:
            for _ in Generator(Smodel, token_manager.vocab, options, batch_input_ids):
                pass
        except StopIteration:
            pass
        embs = Smodel.emb_h
        for layer, emb in zip(args.layers, embs):
            layer_files[layer][i] = emb.squeeze().detach().cpu().numpy().astype(np.float16)
        Smodel.emb_h = []

    print("✅ All embeddings saved as .npy memmap files.")