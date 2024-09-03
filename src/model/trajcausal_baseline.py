import torch
import pickle
import math
import numpy as np
import torch.nn as nn
from base.model import BaseModel
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import random

from model.base_block import *
from model.enc_block import *


class TrajCausal(BaseModel):
    def __init__(self, name, c_in=6, c_env=0, c_out=4, embed_dim=64, n_filters=64, gru_layers=2, base_encoder='gru',
                 qk_dim=64, envbook_num=50, envbook_dim=64,
                 flag_meta=True, cat_or_add='cat', with_random=True, without_point_attention=False, writer=None):
        super(TrajCausal, self).__init__()
        self.name = name
        self.c_in = c_in
        self.c_env = c_env
        self.c_out = c_out
        self.n_filters = n_filters

        self.flag_meta = flag_meta
        self.cat_or_add = cat_or_add
        self.with_random = with_random
        self.without_point_attention = without_point_attention

        self.writer = writer

        self.point_att = None
        self.perplexity = None
        self.env_selects = None

        ###############################################Traj Encoder###############################################
        # 
        self.conv1d_traj = nn.Sequential(nn.Conv1d(c_in+c_env, embed_dim, 3, 1, 1), 
                                         nn.ReLU(inplace=True)
                                         )

        self.object_encoder = UniEncBlock(hidden_size=embed_dim, cell=base_encoder)
        
        self.o_meanblock = TimeMeanBlock()
        
        self.object_readout_layer = ReadoutLayer(embed_dim, c_out)


    def forward(self, _x, m, _len):
        inputs = _x
        ori_inputs = inputs
        # inputs: [bs, n_time, in_channels, 1]      
        # m:      [bs, m_dim]
        inputs = inputs.squeeze()
        envs = inputs.squeeze()[..., 6: 6+self.c_env]
        trajs = inputs.squeeze()[..., :self.c_in]

        dt = inputs[..., 2]
        dd = inputs[..., 3]

        x = torch.cat((trajs, envs), dim=2)
        x = self.conv1d_traj(x.permute(0, 2, 1)).permute(0, 2, 1)     # b, n_time, embed_dim
        
        xo,_ = self.object_encoder(x, dt, dd, _len)
        mean_o = self.o_meanblock(xo, _len)

        out_o_logis = self.object_readout_layer(mean_o)

        return torch.zeros_like(out_o_logis), out_o_logis, torch.zeros_like(out_o_logis)