import numpy as np
import os
import torch
import pickle
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

class MyData(Dataset):
    def __init__(self, normed_traj, metadata, traj, label):
        self.normed_traj = normed_traj
        self.metadata = metadata
        self.traj = traj
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.normed_traj[idx], self.metadata[idx], self.traj[idx], self.label[idx]

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    m = [i[1] for i in data]
    traj = [i[2] for i in data]
    y = [i[3] for i in data]
    data = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    return data.unsqueeze(-1), data_length, torch.stack(m), x, traj, torch.stack(y)

def get_dataloader_dict(datapath, batch_size=64):
    with open(datapath, "rb") as fp:  # Pickling
        data_ori = pickle.load(fp)
        data = {
            'train': [data_ori['X_train'], data_ori['M_train'], data_ori['Y_train']],
            'valid': [data_ori['X_valid'], data_ori['M_valid'], data_ori['Y_valid']],
            'test':  [data_ori['X_test'], data_ori['M_test'], data_ori['Y_test']]
        }
    dataloader_dict = {}

    with open(datapath.split('processed_data')[0]+"scaler.pkl", "rb") as fp:   # Unpickling
        scaler = pickle.load(fp)
    
    if 'geolife' in datapath:
        dataset = 'geolife'
    else:
        dataset = 'grab'

    modes = ['train', 'valid', 'test']
    shuffles = [True, False, False]
    for mode, shuffle in zip(modes, shuffles):
        dataloader_dict[mode] = _get_dataloader(data[mode], shuffle, batch_size, dataset, scaler)
        print('# of intances in {}: {}'.format(mode, len(dataloader_dict[mode].dataset)))
    return dataloader_dict

def _get_dataloader(data, shuffle=True, batch_size=64, dataset='geolife', scaler=None):
    if dataset == 'geolife':
        mode2label = {'walk': 0, 'driving': 1, 'bus': 2, 'bike': 3}
    else:
        mode2label = {'motorcycle': 0, 'car': 1}

    M = [torch.Tensor(m) for m in data[1]]
    T = [torch.Tensor(t) for t in data[0]]
    Y = torch.LongTensor([mode2label[y] for y in data[-1]])

    E = [torch.Tensor(t[:, 6:]) for t in T]
    X = [torch.Tensor(scaler.transform(t[:, :6])) for t in T]
    X = [torch.cat((x, e), dim=1) for x, e in zip(X, E)]

    data = MyData(X, M, T, Y)
    dataloader = DataLoader(data, batch_size=batch_size,
                            shuffle=shuffle, collate_fn=collate_fn)
    return dataloader