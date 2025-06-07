import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from steer.SteerChameleon.rater import LinearProber
from torch.utils.data import Subset
import torch.nn.utils.rnn as rnn_utils
import argparse
import os
import random
import numpy as np




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-l","--emb_layer",type = int,default=24)
    parser.add_argument("-b","--batch_size",type = int, default = 8)
    parser.add_argument("-n","--num_epochs", type = int, default = 100)
    parser.add_argument("--lr",type = float, default=1e-5)
    parser.add_argument("--ratio",type = float, default=0.2) # split ratio for train and val
    parser.add_argument("--embs_dir", type=str, default="", help="Directory for embedding files")
    parser.add_argument("--save_ckpt_dir", type=str, default="", help="Directory to save checkpoints")
    parser.add_argument("--device", type=int, default=1, help="CUDA device number (e.g., 0 for 'cuda:0')")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden size for the model")

    return parser.parse_args()



#for LineearProber
def collate_fn(batch):
    embeddings, labels = zip(*batch)
    return torch.stack(embeddings), torch.stack(labels)

class CombinedDataset(Dataset):
    def __init__(self,dataset1,dataset2):
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_length = len(dataset1) + len(dataset2)
    
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]

class NpyEmbeddingDataset(Dataset):
    def __init__(self, npy_file, toxic=True):
        self.embs = np.load(npy_file, mmap_mode='r')
        self.toxic = toxic

    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, idx):
        emb = torch.tensor(self.embs[idx], dtype=torch.float32)
        label = torch.tensor(0 if self.toxic else 1, dtype=torch.long)
        return emb, label


def split_pair_datasets(pos_dataset, neg_dataset, val_ratio=0.2, seed=42):
    """
    Split the dataset into training and validation sets based on positive and negative samples.

    Parameters:
        pos_dataset: Dataset, the set of positive samples.
        neg_dataset: Dataset, the set of negative samples (same length and order as pos_dataset).
        val_ratio: float, the proportion of data to use for validation.
        seed: int, random seed for shuffling.

    Returns:
        train_dataset: CombinedDataset, the training set (pos + neg).
        val_dataset: CombinedDataset, the validation set (pos + neg).
    """
    assert len(pos_dataset) == len(neg_dataset), "pos and neg dataset must be the same length"

    random.seed(seed)
    num_pairs = len(pos_dataset)
    indices = list(range(num_pairs))
    random.shuffle(indices)

    val_pair_count = int(val_ratio * num_pairs)
    val_indices = indices[:val_pair_count]
    train_indices = indices[val_pair_count:]

    # 构建子集
    pos_train = Subset(pos_dataset, train_indices)
    neg_train = Subset(neg_dataset, train_indices)
    pos_val = Subset(pos_dataset, val_indices)
    neg_val = Subset(neg_dataset, val_indices)

    train_dataset = CombinedDataset(pos_train, neg_train)
    val_dataset = CombinedDataset(pos_val, neg_val)

    return train_dataset, val_dataset


def train(model, train_dataloader, optimizer, criterion, save_dir, layer, num_epochs=3, val_dataloader=None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embs, labels = batch
            embs, labels = embs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(embs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct_preds / total_preds
        print(f"[Train] Epoch {epoch+1} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        # Vliadation
        if val_dataloader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    embs, labels = batch
                    embs, labels = embs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(embs)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_accuracy = val_correct / val_total
            print(f"[Val] Epoch {epoch+1} - Accuracy: {val_accuracy:.4f}")
            model.train()

        os.makedirs(f"{save_dir}/layer{layer}", exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/layer{layer}/epoch_{epoch+1}.pt")
# START TRAINING
if __name__ == "__main__":
    args = parse_args()
    DEVICE = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    layer = args.emb_layer
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.num_epochs

    neg_emb_file = os.path.join(args.embs_dir, "neg", f"{layer}layer.npy")
    pos_emb_file = os.path.join(args.embs_dir, "pos", f"{layer}layer.npy")
    toxic_train_dataset = NpyEmbeddingDataset(neg_emb_file, toxic=True)
    non_toxic_train_dataset = NpyEmbeddingDataset(pos_emb_file, toxic=False)
    train_dataset = CombinedDataset(toxic_train_dataset,non_toxic_train_dataset)


    train_dataset, val_dataset = split_pair_datasets(
        non_toxic_train_dataset,
        toxic_train_dataset,
        val_ratio=args.ratio,
        seed=42
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    rater = LinearProber(DEVICE=DEVICE, checkpoint_pth=None,hidden_sizes=[args.hidden_size,64,2]).to(DEVICE).to(torch.float32)#if float 16, gradient will be nan(vanishing gradient problem)


    optimizer = AdamW(rater.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train(rater, train_dataloader, optimizer, criterion,args.save_ckpt_dir,layer,num_epochs=num_epochs,val_dataloader=val_dataloader)




