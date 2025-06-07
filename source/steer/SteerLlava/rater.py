import torch
import torch.nn as nn


class LinearProber(nn.Module):
    def __init__(self, DEVICE, checkpoint_pth = None, hidden_sizes=[4096, 64, 2]):
        super(LinearProber, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
        if checkpoint_pth is not None:
            self.load_checkpoint(DEVICE,checkpoint_pth)
        else: print("Rater init first time.")
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        #print((self.softmax(x)).shape)
        return self.softmax(x)
    
    def load_checkpoint(self,DEVICE, checkpoint_path):
        """加载指定路径的检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.load_state_dict(checkpoint)
        print(f"Rater loaded from {checkpoint_path}")