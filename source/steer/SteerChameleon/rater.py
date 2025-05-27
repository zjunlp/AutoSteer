import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch.nn as nn
from datasets import load_dataset

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# class text_rater(nn.Module):
#     def __init__(self, checkpoint_path='Roberta_ckpt'):
#         super(text_rater, self).__init__()
#         self.model = RobertaForSequenceClassification.from_pretrained('Roberta_ckpt')
#         #self.model.save_pretrained('Roberta_ckpt')
#         self.fc = nn.Linear(4096, 768)  # Chameleon_emb_dim = 4096, Roberta_emb_dim = 768

#         # 如果传入了checkpoint路径，加载模型参数
#         if checkpoint_path:
#             self.load_checkpoint(checkpoint_path)

#     def load_checkpoint(self, checkpoint_path):
#         """加载指定路径的检查点"""
#         checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#         self.load_state_dict(checkpoint)
#         print(f"Model loaded from {checkpoint_path}")

#     def forward(self, embs, labels=None):
#         emb_bert = self.fc(embs)
#         # print("txt")
#         # print(emb_bert.shape)
#         output = self.model(inputs_embeds=emb_bert, labels=labels)
#         score = output.logits[0,1].tolist()
#         print("txt")
#         return (score,output)

# class vis_rater(nn.Module):
#     def __init__(self, checkpoint_path=None):
#         super(vis_rater, self).__init__()
#         self.model = RobertaForSequenceClassification.from_pretrained('Roberta_ckpt').to(DEVICE)
#         #self.model.save_pretrained('Roberta_ckpt')
#         self.fc = nn.Linear(4096, 768)  # Chameleon_emb_dim = 4096 (Same for vis_emb), Roberta_emb_dim = 768

#         # 如果传入了checkpoint路径，加载模型参数
#         if checkpoint_path:
#             self.load_checkpoint(checkpoint_path)

#     def load_checkpoint(self, checkpoint_path):
#         """加载指定路径的检查点"""
#         checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#         self.load_state_dict(checkpoint)
#         print(f"Model loaded from {checkpoint_path}")

#     def forward(self, embs, labels=None,batch_size =1) :
#         emb_bert = self.fc(embs.reshape(batch_size,embs.shape[0]//batch_size,embs.shape[1]))
#         # print("vis")
#         # print(emb_bert.shape)
#         output = self.model(inputs_embeds=emb_bert, labels=labels)
#         score = output.logits[0,1].tolist()
#         print("vis")
#         return (score,output)
    
# class AutoBalancer:
#     def __init__(self,text_rater_pth,vis_rater_pth,text_score =None,vis_score = None,normalization_factor=10,text_normalization_factor=4,img_normalization_factor = 10,DEVICE="cuda"):
#         self.text_score = text_score
#         self.vis_score = vis_score
#         self.normalization_factor = normalization_factor
#         self.text_normalization_factor = text_normalization_factor
#         self.img_normalization_factor = img_normalization_factor
#         self.text_rater = text_rater(text_rater_pth).to(DEVICE)
#         self.vis_rater = vis_rater(vis_rater_pth).to(DEVICE)
#         self.auto_steer = True
#         self.steer_epsilon = None
        
#     def calculate(self):
#         if(self.text_score is not None):
#             text_score = self.text_score
#         else: text_score = 0
#         if(self.vis_score is not None):
#             vis_score = self.vis_score
#         else: text_score = 0

#         final_score = self.compute(text_score,vis_score)
#         epsilon = final_score/self.normalization_factor

#         return (epsilon,final_score)
        
#     def calculate(self,tok_emb,vis_emb):
#         if(self.auto_steer):
#             text_score = self.text_rater(tok_emb)[0] #idx 0 is the score in tuple (score,outputs)
#             vis_score = self.vis_rater(vis_emb)[0]

#             final_score = self.compute(text_score,vis_score)
#             epsilon = final_score/self.normalization_factor
#             return (epsilon,final_score)
#         else:
#             return (self.steer_epsilon,None)
    
#     def compute(self,text_score,vis_score,method = "avg"):
#         print(f"Txt and Vis scores are: {text_score} and {vis_score}")
#         if method == "avg":
#             text_score/=self.text_normalization_factor
#             vis_score/=self.img_normalization_factor
#             return (text_score*0.6+vis_score*0.4)
#         # other methods could be added
#     def set_autosteer(self,auto=True):
#         self.auto_steer=auto
#     def set_epsilon(self,epsilon=0):
#         self.steer_epsilon=epsilon

# class mlp_rater(nn.Module):
#     def __init__(self,checkpoint_pth = None):
#         super(mlp_rater,self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(4096, 2048),  # 输入 4096, 输出 2048
#             nn.Sigmoid(),              # 激活函数
#             nn.Linear(2048, 1024),  # 输入 2048, 输出 1024
#             nn.Sigmoid(),              # 激活函数
#             nn.Linear(1024, 2)      # 输入 1024, 输出 2
#         )
#         if checkpoint_pth is not None:
#             self.load_checkpoint(checkpoint_pth)
#         else: print("Rater init first time.")

#     def forward(self,input_h):
#         h = self.model(input_h)
#         #print(h)
#         max_values, _ = torch.max(h, dim=1) # pooling; eg. [1, 3]
#         #print(max_values)
#         softmax_output = torch.softmax(max_values ,dim = 1)
#         #prob_0 = softmax_output[0] 
#         return softmax_output # 0 for good, 1 for bad(index 0 for bad, ---> 1 worse)
    
#     def load_checkpoint(self, checkpoint_path):
#         """加载指定路径的检查点"""
#         checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#         self.load_state_dict(checkpoint)
#         print(f"Rater loaded from {checkpoint_path}")

# class mlp_rater_ReLU(nn.Module):
#     def __init__(self,checkpoint_pth = None):
#         super(mlp_rater_ReLU,self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(4096, 64),  # 输入 4096, 输出 2048
#             nn.ReLU(),              # 激活函数
#             nn.Linear(64, 2)      # 输入 1024, 输出 2
#         )
#         if checkpoint_pth is not None:
#             self.load_checkpoint(checkpoint_pth)
#         else: print("Rater init first time.")

#     def forward(self,input_h):
#         h = self.model(input_h)
#         #print(h)
#         max_values, _ = torch.max(h, dim=1) # pooling; eg. [1, 3]
#         #print(max_values)
#         softmax_output = torch.softmax(max_values ,dim = 1)
#         #prob_0 = softmax_output[0] 
#         return softmax_output # 0 for good, 1 for bad(index 0 for bad, ---> 1 worse)
    
#     def load_checkpoint(self, checkpoint_path):
#         """加载指定路径的检查点"""
#         checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#         self.load_state_dict(checkpoint)
#         print(f"Rater loaded from {checkpoint_path}")

class LinearProber(nn.Module):
    def __init__(self, DEVICE,checkpoint_pth = None, hidden_sizes=[4096, 64, 2]):
        super(LinearProber, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
        if checkpoint_pth is not None:
            self.load_checkpoint(checkpoint_pth,DEVICE)
        else: print("Rater init first time.")
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        #print((self.softmax(x)).shape)
        return self.softmax(x)
    
    def load_checkpoint(self, checkpoint_path,DEVICE):
        """加载指定路径的检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.load_state_dict(checkpoint)
        print(f"Rater loaded from {checkpoint_path}")