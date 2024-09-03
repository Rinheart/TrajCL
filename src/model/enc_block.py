import torch
import pickle
import math
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GRUHiddenBlock(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, n_layers=2, dropout=0.2, bidirectional=False):
        super(GRUHiddenBlock, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional)
        
    def forward(self, x_unpacked, x_lens):
        # x_unpacked:  [bs, n_time, in_channels]  2048,100,64  -> [bs, hidden_size]  2048,64
        b, n_time, in_channels = x_unpacked.shape
        x_pack = rnn_utils.pack_padded_sequence(x_unpacked, x_lens, batch_first=True)
        out, h = self.gru(x_pack)  
        out, x_lens = rnn_utils.pad_packed_sequence(out, batch_first=True)    # out: bs, n_time, hidden_size*2

        return out, h


class LSTMHiddenBlock(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, n_layers=2, dropout=0.2, bidirectional=False):
        super(LSTMHiddenBlock, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional)
        
    def forward(self, x_unpacked, x_lens):
        # x_unpacked:  [bs, n_time, in_channels]  2048,100,64  -> [bs, hidden_size]  2048,64
        b, n_time, in_channels = x_unpacked.shape
        x_pack = rnn_utils.pack_padded_sequence(x_unpacked, x_lens, batch_first=True)
        out, h = self.lstm(x_pack)  
        out, x_lens = rnn_utils.pad_packed_sequence(out, batch_first=True)    # out: bs, n_time, hidden_size*2

        return out, h


# UniEncBlock is the same as above GRU/LSTMHiddenBlock
# It can be replaced by any sequence encoder
class UniEncBlock(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, cell='gru'):
        super(UniEncBlock, self).__init__()

        # feel free to implement and replace your enc block
        # self.rnn = RNNEnc(hidden_size=hidden_size, cell=cell)
        # self.rnn = TrajFormerBase(c_in=input_size, c_out=hidden_size, token_dim=hidden_size)

        
    def forward(self, x_unpacked, dt, dd, _len):
        # [bs, n_time, emb] -> [bs, n_time, emb]
        
        out = self.rnn(x_unpacked, dt, dd)

        return out, None


########################### aggregate #############################

class TimeMeanBlock(nn.Module):
    def __init__(self):
        super(TimeMeanBlock, self).__init__()
        
    def forward(self, x_unpacked, x_lens):
        # x_unpacked:  [bs, n_time, hidden_size]  2048,100,64  -> [bs, hidden_size]  2048,64
        x_lens = x_lens.to(x_unpacked.device)
        mean_out = x_unpacked.sum(dim=1).div(x_lens.float().unsqueeze(dim=1))    # mean_out: bs, hidden_size
        return mean_out
    

class TimeLastBlock(nn.Module):
    def __init__(self):
        super(TimeLastBlock, self).__init__()
        
    def forward(self, x_unpacked, x_lens):
        # x_unpacked:  [bs, n_time, hidden_size]  2048,100,64  -> [bs, hidden_size]  2048,64
        out = x_unpacked
        indices = torch.LongTensor(np.array(x_lens) - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1).to(out.device)
        last_encoded_outs = out.gather(dim=1, index=indices).squeeze(dim=1)
        return last_encoded_outs


class TimePoolBlock(nn.Module):
    def __init__(self):
        super(TimePoolBlock, self).__init__()
        self.adapool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x_unpacked, x_lens):
        x = self.adapool(x_unpacked.permute(0, 2, 1)).squeeze()
        return x