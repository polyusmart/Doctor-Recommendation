import os
import torch
from torch import nn

EMBED_SIZE = 768

class MultiHeadAtt(nn.Module):
    def __init__(self, head_num, dropout):
        super(MultiHeadAtt, self).__init__()
        self.hid_size = EMBED_SIZE
        self.head_num = head_num
        self.head_dim = self.hid_size // self.head_num

        self.W_Q = nn.Linear(self.hid_size, self.hid_size)
        self.W_K = nn.Linear(self.hid_size, self.hid_size)
        self.W_V = nn.Linear(self.hid_size, self.hid_size)

        self.fc = nn.Linear(self.hid_size, self.hid_size)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        self.scale = self.scale.cuda()

    def forward(self, query, key, value, scale=None):
        batch_size = query.shape[0]
        
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        Q = Q.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        if scale == None:
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        else:
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_size)
        x = self.fc(x)
        return x, attention
