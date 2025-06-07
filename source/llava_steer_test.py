import os
from steer.SteerLlava.steer_model_llava_inference import SteerModelForConditionalGeneration,Eval_Llava
from steer.SteerLlava.constants import model_path, STEER_MATRIX_PATH, INFERENCE_STEER_MATRIX_PATH
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration,LlavaOnevisionConfig
import torch
from PIL import Image
from typing import List, Dict, Tuple
import jsonlines
import argparse
from distutils.util import strtobool 
from tqdm import tqdm
from steer.SteerLlava.rater import LinearProber

cache_path = model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llava Model Inference")
    # Add command-line arguments
    parser.add_argument('--rater_ckpt', type=str, default='/mnt/16t/lyucheng/ANOLE/anole/training/rater_ckpt9/comb_model_epoch_10.pt', help='Output file path')
    parser.add_argument('--epsilon', type=float, default=0.1, help='The epsilon parameter for the model')
    parser.add_argument('--auto', type=str, default="False", help='Whether to use auto mode (True or False)')    
    parser.add_argument('--output_file', type=str, default="", help='Output file path')
    parser.add_argument('--layer', type=int, default=24)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--test_dataset', type=str, default='/mnt/16t/lyucheng/ANOLE/dataset/ToViLaG/Mono_NontoxicText_ToxicImg_500Samples_convert4test2.jsonl', help='Output file path')
    parser.add_argument('--img_dir', type=str, default='COCO')    
    parser.add_argument('--get_toxicity', type=str,default="False")    

    args = parser.parse_args()
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    config = LlavaOnevisionConfig.from_pretrained(cache_path+"/config")
    steer_config = {
        "epsilon": args.epsilon,
        "gen": False,
        "probe_layer": args.layer,
        "need_rating": bool(strtobool(args.auto)),
        "threshhold": 0.5,
        "safety_ratio": 1.0,
        "DEVICE" :args.device,
        "rater_ckpt_pth": args.rater_ckpt,
        "get_toxicity": bool(strtobool(args.get_toxicity))
    }
    config.text_config.steer_config = steer_config

    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir=model_path)

    # Load the model
    model = SteerModelForConditionalGeneration.from_pretrained(
        "llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir=cache_path, torch_dtype=torch.float16, config=config
    ).to(DEVICE)

    if steer_config["need_rating"]:
        model.language_model.model.rater = LinearProber(DEVICE=DEVICE,checkpoint_pth= args.rater_ckpt,hidden_sizes=[3584,64,2]).to(DEVICE).to(dtype=torch.float16)#必须加，不然rater加载的有问题；问题暂不明确

    model.eval()

    eval = Eval_Llava(processor=processor)
    dataset = eval.get_unified_examine_data(pth = args.test_dataset, img_dir_pth= args.img_dir)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    writer = jsonlines.open(args.output_file, mode='w')
    
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        input_data = eval.process_llava_input(sample).to(DEVICE, torch.float16)
        model.set_first_pass()
        
        try:
            output = model.generate(**input_data, max_new_tokens=150, pad_token_id=151645)#151645, open-end generation padding
            output_str = processor.decode(output[0], skip_special_tokens=True, pad_token_id=151645)
            answer = output_str.split("assistant", 1)[1].strip() if "assistant" in output_str else output_str.strip()
        except StopIteration:
            answer = ""

        query, gt_answer = "", None
        for item in sample:
            if item["type"] == "text":
                query += item["content"]
            elif item["type"] == "answer":
                gt_answer = item["content"]

        # Construct the output
        line = {
            "query": query,
            "model_ans": answer
        }
        if gt_answer is not None:
            line["answer"] = gt_answer

        # Add toxicity field if required(converted to float)
        if steer_config["get_toxicity"]:
            toxicity = model.language_model.model.toxicity
            if isinstance(toxicity, torch.Tensor):
                toxicity = float(toxicity.item())
            else:
                toxicity = float(toxicity)
            line["toxicity"] = toxicity

        writer.write(line)

    writer.close()
    print(f"Results saved to {args.output_file}")
    print("Evaluation completed.")

