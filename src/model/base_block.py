import torch
import pickle
import math
import numpy as np
import torch.nn as nn
from base.model import BaseModel
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from timm.models.layers import trunc_normal_


class BaseSegmenter(nn.Module):
    def __init__(self):
        super(BaseSegmenter, self).__init__()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


class EnvSegmenterDualOutput(BaseSegmenter):
    def __init__(self, in_channels, out_channels):
        super(EnvSegmenterDualOutput, self).__init__()
        self.embed_dim1 = int(in_channels*2)
        self.embed_dim2 = int(in_channels*4)

        # Contracting path
        self.conv1 = self.conv_block(in_channels, self.embed_dim1)
        self.down1 = nn.MaxPool1d(2)
        self.conv2 = self.conv_block(self.embed_dim1, self.embed_dim2)

        # Expanding path
        # self.up2 = nn.ConvTranspose1d(self.embed_dim2, self.embed_dim1, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose1d(self.embed_dim2, self.embed_dim1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = self.conv_block(self.embed_dim2, self.embed_dim1)

        # Output layer
        self.convEnv = nn.Conv1d(self.embed_dim1, out_channels, kernel_size=1)
        self.convObj = nn.Conv1d(self.embed_dim1, out_channels, kernel_size=1)

    def forward(self, x):
        # x is the input tensor of shape [batch_size, channels, sequence_length]

        # Contracting path
        x1 = self.conv1(x)  
        p1 = self.down1(x1)  
        x2 = self.conv2(p1)  

        # Expanding path
        u1 = self.up2(x2)  
        m1 = torch.cat([u1, x1], dim=1)  
        x3 = self.conv3(m1) 

        # Output layer
        outEnv = torch.sigmoid(self.convEnv(x3))
        outObj = torch.sigmoid(self.convObj(x3))

        return outEnv, outObj



class VQCrossAttenSeg(nn.Module):
    def __init__(self, in_channels, qk_dim, envbook_num, envbook_dim):
        super(VQCrossAttenSeg, self).__init__()
        self.in_channels = in_channels
        self.qk_dim = qk_dim
        self.envbook_num = envbook_num
        self.envbook_dim = envbook_dim

        self.scaling = qk_dim ** -0.5
        
        # Learnable embedding (n*d)
        self.env_codebook = nn.Parameter(torch.randn(envbook_num, envbook_dim))
        # trunc_normal_(self.env_codebook, std=.02)
        
        # Query, key, value projections for attention
        self.query_proj = nn.Linear(in_channels, qk_dim, bias=False)
        self.key_proj = nn.Linear(envbook_dim, qk_dim, bias=False)
        self.value_proj = nn.Linear(envbook_dim, 1, bias=False)

    def forward(self, x):
        # x is the input tensor of shape [batch_size, sequence_length, in_channels]
        
        # Project trajectories to query, and embeddings to key and value

        query = self.query_proj(x)
        key = self.key_proj(self.env_codebook)
        value = self.value_proj(self.env_codebook)
        
        # Compute attention scores
        # Shape: [batch_size, sequence_length, num_env]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling  

        # Sample from the Gumbel-Softmax distribution to get hard assignments
        env_selects = F.gumbel_softmax(attention_scores, hard=True, dim=-1)
        
        # Optionally multiply the probabilities with the value embeddings
        # This step depends on how you want to use the selected embeddings
        env_att = torch.matmul(env_selects, value)

        env_att = torch.sigmoid(env_att)
        
        return env_att, env_selects
    

class VQCrossAttenSeg_stable(nn.Module):
    def __init__(self, in_channels, qk_dim, envbook_num, envbook_dim):
        super(VQCrossAttenSeg_stable, self).__init__()
        self.in_channels = in_channels
        self.qk_dim = qk_dim
        self.envbook_num = envbook_num
        self.envbook_dim = envbook_dim

        self.scaling = qk_dim ** -0.5
        
        # Learnable embedding (n*d)
        self.env_codebook = nn.Parameter(torch.randn(envbook_num, envbook_dim))
        # trunc_normal_(self.env_codebook, std=.02)
        
        # Query, key, value projections for attention
        self.query_proj = nn.Linear(in_channels, qk_dim, bias=False)
        self.key_proj = nn.Linear(envbook_dim, qk_dim, bias=False)
        self.value_proj = nn.Linear(envbook_dim, 1, bias=False)

    def forward(self, x):
        # x is the input tensor of shape [batch_size, sequence_length, in_channels]
        
        # stable version--randomly initialize the codebook
        env_codebook = torch.randn(self.envbook_num, self.envbook_dim, requires_grad=False).to(x.device)
        
        # Project trajectories to query, and embeddings to key and value
        query = self.query_proj(x)
        key = self.key_proj(env_codebook)
        value = self.value_proj(env_codebook)
        
        # Compute attention scores
        # Shape: [batch_size, sequence_length, num_env]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling  

        # Sample from the Gumbel-Softmax distribution to get hard assignments
        env_selects = F.gumbel_softmax(attention_scores, hard=True, dim=-1)
        
        # Optionally multiply the probabilities with the value embeddings
        # This step depends on how you want to use the selected embeddings
        env_att = torch.matmul(env_selects, value)

        env_att = torch.sigmoid(env_att)
        
        return env_att, env_selects



class VQCodebookSeg_multihead(nn.Module):
    def __init__(self, in_channels, qk_dim, envbook_num, envbook_dim, num_heads=1):
        super(VQCodebookSeg_multihead, self).__init__()
        self.in_channels = in_channels
        self.qk_dim = qk_dim
        self.envbook_num = envbook_num+1
        self.envbook_dim = envbook_dim
        self.num_heads = num_heads
        self.per_head_dim = qk_dim // num_heads

        self.scaling = qk_dim ** -0.5
        
        # Learnable embedding (n+1)*d
        self.env_codebook = nn.Parameter(torch.randn(self.envbook_num, self.envbook_dim))
        # trunc_normal_(self.env_codebook, std=.02)
        
        # Query, key, value projections for attention
        self.query_proj = nn.Linear(in_channels, qk_dim, bias=False)
        self.key_proj = nn.Linear(envbook_dim, qk_dim, bias=False)

    def forward(self, x):
        # x is the input tensor of shape [batch_size, sequence_length, in_channels]
        batch_size, seq_length, _ = x.size()
        
        # Project trajectories to query, and embeddings to key and value
        query = self.query_proj(x)
        key = self.key_proj(self.env_codebook)

        query = self.query_proj(x).view(batch_size, seq_length, self.num_heads, self.per_head_dim)
        key = self.key_proj(self.env_codebook).view(self.envbook_num, self.num_heads, self.per_head_dim)
        query = query.transpose(1, 2).contiguous()  # [batch_size, num_heads, seq_length, per_head_dim]
        key = key.transpose(0, 1).contiguous()      # [num_heads, envbook_num, per_head_dim]
        
        # Compute attention scores
        # Shape: [batch_size, sequence_length, num_env]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling  

        attention_probs = F.softmax(attention_scores, dim=-1)   # bs, seq_len, env_num+1
        
        causal_att = attention_probs[...,0]                     # bs, seq_len
        # here is soft select
        env_selects = attention_probs[..., 1:]                  
        env_att = env_selects.sum(dim=-1)                       # bs, seq_len
        
        return env_att, causal_att, env_selects


class ReadoutLayer(nn.Module):
    def __init__(self, embed_dim, c_out):
        super(ReadoutLayer, self).__init__()
        self.fc1_bn = nn.BatchNorm1d(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2_bn = nn.BatchNorm1d(embed_dim)
        self.fc2 = nn.Linear(embed_dim, c_out)
        
    def forward(self, x):
        x = self.fc1_bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2_bn(x)
        x = self.fc2(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class RandomReadoutLayer(nn.Module):
    def __init__(self, embed_dim, c_out, cat_or_add, with_random):
        super(RandomReadoutLayer, self).__init__()
        self.with_random = with_random
        self.cat_or_add = cat_or_add

        # Random mlp - do(x) Causal intervention
        if self.cat_or_add == "cat":
            self.fc1_bn_co = nn.BatchNorm1d(embed_dim * 2)
            self.fc1_co = nn.Linear(embed_dim * 2, embed_dim)
        elif self.cat_or_add == "add":
            self.fc1_bn_co = nn.BatchNorm1d(embed_dim)
            self.fc1_co = nn.Linear(embed_dim, embed_dim)
        else:
            assert False
        self.fc2_bn_co = nn.BatchNorm1d(embed_dim)
        self.fc2_co = nn.Linear(embed_dim, c_out)
        
    def forward(self, xc, xo):

        num = xc.shape[0]                   # bs
        l = [i for i in range(num)]
        if self.with_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)        
        if self.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)   # bs, xc_feat+xo_feat (2*embed_dim)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        # x_logis = F.log_softmax(x, dim=-1)
        # return x_logis
        return x
