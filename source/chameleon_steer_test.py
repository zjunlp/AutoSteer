import os
import sys
import argparse
import json
import inspect
import glob
from pathlib import Path
from distutils.util import strtobool
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
import jsonlines

from datasets import load_dataset
from steer.SteerChameleon.constants import (
    TOKENIZER_TEXT_PATH,
    TOKENIZER_IMAGE_CFG_PATH,
    TOKENIZER_IMAGE_PATH,
    MODEL_7B_PATH,
    ANOLE_DIR_PATH
)
sys.path.append(ANOLE_DIR_PATH)
from chameleon.inference.chameleon import ChameleonInferenceModel, Options, TokenManager, Generator
from chameleon.inference.transformer import ModelArgs
from steer.SteerChameleon.steer_model_chameleon_inference import SteerModel, Eval_Chameleon

# # Set default model and file paths
# tmr_pth = "/mnt/16t/lyucheng/ANOLE/Anole-7b-v0.1/models/7b"

# Load model from consolidated weights and additional parameters
def load_steer_model(path: str,
                     epsilon: float,
                     auto: bool,
                     rater_ckpt_pth: str,
                     safety_ratio: float,
                     device: str,
                     selected_layer: int,
                     get_toxicity = bool,
                     rank: int = 0,
                     gen: bool = False) -> SteerModel:
    src_dir = Path(path)

    with open(src_dir / "params.json", "r") as f:
        params = json.loads(f.read())
    with open(src_dir / "consolidate_params.json", "r") as f:
        consolidate_params = json.loads(f.read())
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
        DEVICE="cpu",
        gen=gen,
        get_toxicity = get_toxicity
    )

    ckpt_path = _get_checkpoint_path(src_dir, rank)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state, strict=False)

    model.DEVICE = device
    model = model.to(torch.float16).to(device)
    model.selected_layer = selected_layer
    return model

# Determine checkpoint file path
def _get_checkpoint_path(src_dir: Path, rank: int | None) -> Path:
    base_path = src_dir / "consolidated.pth"
    if not rank and base_path.exists():
        return base_path

    alt_path = src_dir / f"consolidated.{rank:02}.pth"
    if alt_path.exists():
        return alt_path

    raise ValueError("Consolidated checkpoint not found.")

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Chameleon Model Inference")
    parser.add_argument('--rater_ckpt', type=str,default="", help='Path to rater checkpoint')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for steering')
    parser.add_argument('--auto', type=str, default="False", help='Enable auto mode (True/False)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save output results')
    parser.add_argument('--layer', type=int, default=24, help='Layer to steer')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--test_dataset', type=str, required=True, help='Path to JSONL test dataset')
    parser.add_argument('--img_dir', type=str, default='COCO', help='Directory containing image files')
    parser.add_argument('--output_pic_dir', type=str, default=None, help='Optional output directory for images')
    parser.add_argument('--get_toxicity', type=str, default="False")

    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    args = parse_args()
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    epsilon = args.epsilon
    auto = bool(strtobool(args.auto))
    get_toxicity = bool(strtobool(args.get_toxicity))
    output_file = args.output_file
    output_pic_dir = args.output_pic_dir

    steer_model_kwargs = {
        "path": MODEL_7B_PATH,
        "epsilon": epsilon,
        "auto": auto,
        "rater_ckpt_pth": args.rater_ckpt,
        "safety_ratio": 1,
        "device": DEVICE,
        "selected_layer": args.layer,
        "get_toxicity": get_toxicity,
    }

    # Load steerable model
    Smodel = load_steer_model(**steer_model_kwargs)

    # Load and process dataset
    eval = Eval_Chameleon()
    dataset = eval.get_unified_examine_data(pth=args.test_dataset, img_dir_pth=args.img_dir)

    # Initialize tokenizer
    token_manager = TokenManager(
        tokenizer_path=TOKENIZER_TEXT_PATH.as_posix(),
        vqgan_cfg_path=TOKENIZER_IMAGE_CFG_PATH.as_posix(),
        vqgan_ckpt_path=TOKENIZER_IMAGE_PATH.as_posix(),
        device=DEVICE,
    )

    torch.set_default_device(DEVICE)
    torch.cuda.set_device(args.device)

    options = Options()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with jsonlines.open(output_file, mode='w') as writer:
        for i in tqdm(range(len(dataset)), desc="Running Inference"):
            input = dataset[i]
            try:
                input = eval.process_chameleon_input(input)
                batch_input_ids = [
                    token_manager.tokens_from_ui(prompt_ui)
                    for prompt_ui in input
                ]

                output = []
                Smodel.first_pass = True
                for token in Generator(
                    model=Smodel,
                    vocab=token_manager.vocab,
                    options=options,
                    input_ids=batch_input_ids,
                ):
                    output.append(token)

                output_ids = [t.id for t in output]
                output_ids = torch.stack(output_ids).T
                string_outputs = eval.process_chameleon_output_(token_manager, output_ids, output_pic_dir, i)

                # 构建输出
                query = ""
                ans = None
                for segment in dataset[i]:
                    if segment['type'] == 'text':
                        query += segment['content']
                    elif segment['type'] == 'answer':
                        ans = segment['content']

                record = {'query': query, 'model_ans': string_outputs}
                if ans is not None:
                    record['answer'] = ans
                if get_toxicity:
                    toxicity = Smodel.toxicity
                    if isinstance(toxicity, torch.Tensor):
                        toxicity = float(toxicity.item())
                    else:
                        toxicity = float(toxicity)
                    record['toxicity'] = toxicity

                writer.write(record)

            except Exception as e:
                print(f"[Warning] Error occurred at index {i}: {e}")
                print("[Info] Retrying once...")

                try:
                    torch.cuda.empty_cache()
                    input = dataset[i]
                    input = eval.process_chameleon_input(input)
                    batch_input_ids = [
                        token_manager.tokens_from_ui(prompt_ui)
                        for prompt_ui in input
                    ]

                    output = []
                    Smodel.first_pass = True
                    for token in Generator(
                        model=Smodel,
                        vocab=token_manager.vocab,
                        options=options,
                        input_ids=batch_input_ids,
                    ):
                        output.append(token)

                    output_ids = [t.id for t in output]
                    output_ids = torch.stack(output_ids).T
                    string_outputs = eval.process_chameleon_output_(token_manager, output_ids, output_pic_dir, i)

                    # 构建输出
                    query = ""
                    ans = None
                    for segment in dataset[i]:
                        if segment['type'] == 'text':
                            query += segment['content']
                        elif segment['type'] == 'answer':
                            ans = segment['content']

                    record = {'query': query, 'model_ans': string_outputs}
                    if ans is not None:
                        record['answer'] = ans
                    if get_toxicity:
                        toxicity = Smodel.toxicity
                        if isinstance(toxicity, torch.Tensor):
                            toxicity = float(toxicity.item())
                        else:
                            toxicity = float(toxicity)
                        record['toxicity'] = toxicity

                    writer.write(record)
                except Exception as e2:
                    print(f"[Error] Retry failed at index {i}: {e2}")
                    continue  # 跳过该条

    print(f"Results saved to {output_file}")
    print("Evaluation completed.")