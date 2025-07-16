import os
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

def load_caa(pos_path, neg_path, sample_num):
    """
    Load .npy embeddings, randomly sample indices, and compute CAA = pos - neg
    """
    pos = np.load(pos_path, mmap_mode='r')
    neg = np.load(neg_path, mmap_mode='r')
    assert pos.shape == neg.shape, "Shape mismatch between pos and neg"
    assert sample_num <= pos.shape[0], "sample_num too large"
    idx = np.random.choice(pos.shape[0], size=sample_num, replace=False)
    return pos[idx] - neg[idx]


def visualize_heatmap(caa, layer, output, vmin, vmax):
    """
    Compute cosine-similarity matrix and plot as heatmap with fixed scale.
    """
    os.makedirs(os.path.dirname(output), exist_ok=True)
    sim_matrix = cosine_similarity(caa)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        sim_matrix, cmap="viridis_r",
        vmin=vmin, vmax=vmax,  # 固定 color scale
        cbar=True, xticklabels=False, yticklabels=False
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    plt.title(f"Layer {layer} CAA Cosine Similarity", fontsize=18)

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cosine-similarity heatmap of CAA vectors"
    )
    parser.add_argument(
        "--pos_dir",
        default="/mnt/16t/lyucheng/AutoSteer/source/LayerSelect/embs_llava/pos",
        help="Directory of positive .npy files"
    )
    parser.add_argument(
        "--neg_dir",
        default="/mnt/16t/lyucheng/AutoSteer/source/LayerSelect/embs_llava/neg",
        help="Directory of negative .npy files"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="Layer index (e.g. 4,8,…,24)"
    )
    parser.add_argument(
        "--sample_num", type=int, default=200,
        help="Number of vectors to sample (≤200 recommended for readability)"
    )
    parser.add_argument(
        "--output_dir", default=None, help="Output image path (PNG)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    all_sim_values = []

    for layer in args.layers:
        pos_path = os.path.join(args.pos_dir, f"{layer}layer.npy")
        neg_path = os.path.join(args.neg_dir, f"{layer}layer.npy")
        caa = load_caa(pos_path, neg_path, args.sample_num)
        sim_matrix = cosine_similarity(caa)
        all_sim_values.append(sim_matrix)

    all_sim_concat = np.concatenate([s.flatten() for s in all_sim_values])
    global_vmin, global_vmax = all_sim_concat.min(), all_sim_concat.max()

    for layer, sim_matrix in zip(args.layers, all_sim_values):
        out_path = os.path.join(args.output_dir, f"layer_{layer}_heatmap.png")
        visualize_heatmap(caa=sim_matrix, layer=layer, output=out_path, vmin=global_vmin, vmax=global_vmax)
