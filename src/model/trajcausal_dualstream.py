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
        # self.conv1d_traj = nn.Sequential(nn.Conv1d(c_in+c_env, embed_dim, 3, 1, 1), nn.ReLU(inplace=True))

        self.conv1d_traj = nn.Sequential(nn.Conv1d(c_in+c_env, embed_dim, 3, 1, 1), 
                                         nn.ReLU(inplace=True),
                                         nn.Conv1d(embed_dim, embed_dim, 3, 1, 1)
                                         )

        self.env_bn = nn.BatchNorm1d(c_env)


        ###############################################Environment Extracter###############################################
        # 
        if not without_point_attention:
            # self.conv1d_env = nn.Sequential(nn.Conv1d(c_env, 64, 3, 1, 1), nn.ReLU(inplace=True))
            # self.env_encoder = EnvSegmenter(64, 1)
            # self.VQenv_encoder = VQEnvSegmenter(64, 1, 50, 0.25)
            self.VQcrossattn = VQCrossAttenSeg(c_env, qk_dim, envbook_num, envbook_dim)
            
        ###############################################Context ENC and Object ENC###############################################
        # 
        self.context_encoder = UniEncBlock(hidden_size=embed_dim, cell=base_encoder)
        self.object_encoder = UniEncBlock(hidden_size=embed_dim, cell=base_encoder)

        self.c_meanblock = TimePoolBlock()
        self.o_meanblock = TimePoolBlock()

        self.context_readout_layer = ReadoutLayer(embed_dim, c_out)
        self.object_readout_layer = ReadoutLayer(embed_dim, c_out)
        self.random_readout_layer = RandomReadoutLayer(embed_dim, c_out, cat_or_add, with_random)



    def forward(self, _x, m, _len):
        # inputs, _len = rnn_utils.pad_packed_sequence(x_packed, batch_first=True)
        inputs = _x
        ori_inputs = inputs
        inputs = inputs.squeeze()                   # lat lon timedelta dis v a + n envs
        envs = inputs.squeeze()[..., 6: 6+self.c_env]
        trajs = inputs.squeeze()[..., :self.c_in]   
        
        dt = inputs[..., 2]
        dd = inputs[..., 3]

        b, n_time, _ = trajs.shape

        envs = self.env_bn(envs.permute(0,2,1)).permute(0,2,1)

        ############################################### env input ###############################################
        # only basicfeatures
        if self.c_env==0:
            x = trajs
            e = trajs

        # env=x=allfeatures
        # elif self.c_env!=0:
        #     x = torch.cat((trajs, envs), dim=2)
        #     e = x.clone()

        # dualstream
        elif self.c_env!=0:
            x = torch.cat((trajs, envs), dim=2)
            e = envs

        ############################################### input proj ###############################################

        x = self.conv1d_traj(x.permute(0, 2, 1)).permute(0, 2, 1)     # b, n_time, embed_dim
        
        ############################################### env extractor ###############################################
        if self.without_point_attention:
            point_att = torch.zeros(b, n_time, 1).to(inputs.device)   # context

            point_att_inv = 1 - point_att
            point_att = torch.cat((point_att, point_att_inv), dim=-1)
            
        else:
            point_att, env_selects = self.VQcrossattn(e)
            self.env_selects = env_selects
            point_att_inv = 1 - point_att
            point_att = torch.cat((point_att, point_att_inv), dim=-1)

        # for tensorboard record
        self.point_att = point_att

        ############################################### intervention ###############################################

        xo,_ = self.object_encoder(x, dt, dd, _len)
        xc,_ = self.context_encoder(x, dt, dd, _len)

        xc = point_att[:, :, 0].unsqueeze(-1) * xc  # b, n_time, embed_dim
        xo = point_att[:, :, 1].unsqueeze(-1) * xo  # b, n_time, embed_dim
        
        mean_c = self.c_meanblock(xc, _len)
        mean_o = self.o_meanblock(xo, _len)
        
        # intervention
        out_co = self.random_readout_layer(mean_c, mean_o)
        # read_out
        out_c_logis = self.context_readout_layer(mean_c)
        out_o_logis = self.object_readout_layer(mean_o)

        return out_c_logis, out_o_logis, out_co