import sys
import copy
import torch
import torch.nn.functional as F
from torch import nn
sys.path.append('.')
from .MultiHeadAtt import MultiHeadAtt

class MLP(nn.Module):
    def __init__(self, in_size, hid_size, dropout, head_num, add_self_att_on):
        super(MLP, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.add_self_att_on = add_self_att_on
        if add_self_att_on == "profile":
            print("Model: MUL-ATT W/O D")
            self.goodat_attention = MultiHeadAtt(head_num, dropout)
        elif add_self_att_on == "dialogs":
            print("Model: MUL-ATT W/O P")
            self.dialog_attention = MultiHeadAtt(head_num, dropout)
        else:
            print("Model: MUL-ATT FULL")
            self.attention = MultiHeadAtt(head_num, dropout)
        self.relu= nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=self.hid_size)
        self.fc1 = nn.Linear(self.in_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, rcd_num):
        profile = x[:, 0]
        profile = profile.reshape((profile.shape[0], 1, profile.shape[-1]))
        dialogs = x[:, 1: 1 + rcd_num]
        # our ablations: MUL-ATT (W/O D or W/O P)
        if self.add_self_att_on == "profile":
            q, k, v = copy.deepcopy(profile), copy.deepcopy(profile), copy.deepcopy(profile)
            dr_emb, _ = self.goodat_attention(q, k, v)
        elif self.add_self_att_on == "dialogs":
            q, k, v = copy.deepcopy(dialogs), copy.deepcopy(dialogs), copy.deepcopy(dialogs)
            dr_emb, _ = self.dialog_attention(q, k, v)
            dr_emb = torch.mean(dr_emb, dim=1).unsqueeze(1)
        else: # our model, MUT-ATT (FULL)
            q, k, v = copy.deepcopy(profile), copy.deepcopy(dialogs), copy.deepcopy(dialogs)
            dr_emb, _ = self.attention(q, k, v)
        dr_emb = dr_emb.reshape((dr_emb.shape[0], dr_emb.shape[-1]))
        features = torch.cat((dr_emb, x[:, -1]), 1)
        features = self.relu(self.bn1(self.fc1(features)))
        output = self.fc2(features)
        output = self.sigmoid(output)
        return output