from utils import SafetyAwareness
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_dir", default="/mnt/16t/lyucheng/AutoSteer/source/LayerSelect/embs_llava/pos")
    parser.add_argument("--neg_dir", default="/mnt/16t/lyucheng/AutoSteer/source/LayerSelect/embs_llava/neg")
    parser.add_argument("--sample_num", type=int, default=500)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    layers = [*range(4, 28, 4)]  # llava has 4, 8, ..., 24 layers

    for layer in layers:
        pos_path = f"{args.pos_dir}/{layer}layer.npy"
        neg_path = f"{args.neg_dir}/{layer}layer.npy"

        evaluator = SafetyAwareness(pos_path, neg_path)
        evaluator.random_select(sample_num=args.sample_num)
        score = evaluator.calculate_safetyAwareness()

        print(f"Layer {layer}: Safety Awareness Score = {score:.4f}")