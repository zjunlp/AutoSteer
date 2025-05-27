import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from steer.SteerChameleon.rater import LinearProber
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-layer Binary Evaluation on NPY Embeddings")
    parser.add_argument("--model_ckpt_dir", type=str, required=True, help="Root directory of saved model checkpoints")
    parser.add_argument("--neg_embs_dir", type=str, help="Directory containing negative (toxic) .npy embedding files")
    parser.add_argument("--pos_embs_dir", type=str, help="Directory containing positive (non-toxic) .npy embedding files")
    parser.add_argument("--layers", type=int, nargs="+", required=True, help="List of embedding layers to evaluate")
    parser.add_argument("--epochs", type=int, nargs="+", required=True, help="List of epochs to evaluate")
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()

class NpyEmbeddingDataset(Dataset):
    def __init__(self, npy_file, label: int):
        self.embs = np.load(npy_file, mmap_mode='r')  # Lazy load
        self.label = label

    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, idx):
        emb = torch.tensor(self.embs[idx], dtype=torch.float16)  # Always use float16
        label = torch.tensor(self.label, dtype=torch.long)
        return emb, label

def collate_fn(batch):
    embs, labels = zip(*batch)
    return torch.stack(embs), torch.stack(labels)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embs, labels in dataloader:
            embs, labels = embs.to(device), labels.to(device)
            outputs = model(embs)
            # print("------------------")
            # print(outputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

if __name__ == "__main__":
    args = parse_args()
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    print(f"{'Epoch':>5} | {'Layer':>6} | {'Accuracy':>8}")
    print("-" * 28)

    for epoch in args.epochs:
        for layer in args.layers:
            ckpt_path = os.path.join(args.model_ckpt_dir, f"layer{layer}", f"epoch_{epoch}.pt")
            neg_path = os.path.join(args.neg_embs_dir, f"{layer}layer.npy") if args.neg_embs_dir else None
            pos_path = os.path.join(args.pos_embs_dir, f"{layer}layer.npy") if args.pos_embs_dir else None

            if not os.path.exists(ckpt_path) or (not neg_path and not pos_path):
                print(f"{epoch:>5} | {layer:>6} | {'MISSING':>8}")
                continue

            datasets = []
            if neg_path and os.path.exists(neg_path):
                datasets.append(NpyEmbeddingDataset(neg_path, label=0))
            if pos_path and os.path.exists(pos_path):
                datasets.append(NpyEmbeddingDataset(pos_path, label=1))

            if not datasets:
                print(f"{epoch:>5} | {layer:>6} | {'EMPTY':>8}")
                continue

            full_dataset = ConcatDataset(datasets)
            dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

            model = LinearProber(DEVICE=DEVICE, checkpoint_pth=ckpt_path, hidden_sizes=[args.hidden_size, 64, 2]).to(DEVICE).to(dtype=torch.float16)
            model.eval()
            acc = evaluate(model, dataloader, DEVICE)
            print(f"{epoch:>5} | {layer:>6} | {acc:.4f}")