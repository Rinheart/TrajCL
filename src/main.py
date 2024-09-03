import sys
sys.path.append("..")

import torch
import numpy as np
import os
import time
import argparse
import yaml
import json
import torch.nn as nn
import pandas as pd
from utils.helper import get_dataloader_dict
from torch.utils.tensorboard import SummaryWriter

from trainer.causal_trainer import CausalTrainer

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model.trajcausal_baseline import TrajCausal as TrajCausalBase
from model.trajcausal_dualstream import TrajCausal as TrajCausalFinal

def get_config():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='l-bj-dual-gru')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to run')
    parser.add_argument('--base_encoder', type=str, default='gru')
    
    # model
    parser.add_argument('--input_dim', type=int, default=3)   # lat lon timedelta dis v a
    parser.add_argument('--env_dim', type=int, default=24)    # 0=deactivate---last n dim: road_n=13   full=24
    parser.add_argument('--embed_dim', type=int, default=64)  # total network emb dim
    parser.add_argument('--gru_layers', type=int, default=1)  # gru encoder layers

    # envsegmenter
    parser.add_argument('--latent_dim', type=int, default=64)  # qk dim
    parser.add_argument('--envbook_num', type=int, default=50) # book num
    parser.add_argument('--envbook_dim', type=int, default=64) # book dim

    # causal option
    parser.add_argument('--cat_or_add', type=str, default='add')
    parser.add_argument('--with_random', type=str2bool, default=True)

    parser.add_argument('--o', type=float, default=1.0)     # object:  origin strength
    parser.add_argument('--c', type=float, default=0.5)     # context: disentanglement strength
    parser.add_argument('--co', type=float, default=0.5)    # causal intervention strength
    
    # training
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=35)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)


    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--n_exp', type=int, default=1,help='experiment index')
    parser.add_argument('--seed', type=int, default=2017)

    augs = parser.parse_args()
    print(augs)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(augs.gpu)
    return augs


def main():
    augs = get_config()

    if 'bj' in augs.model_name:
        dataloader_dict = get_dataloader_dict(os.path.join('./trajdata', 'geolife', 'processed_data', 'x.pkl'), 256)
        output_dim = 4
        augs.log_dir = './logs/bj/{}/'.format(augs.model_name)

    n_filters = 64
    flag_meta = False

    augs_str = '  \n'.join(f'{k}: {v}' for k, v in vars(augs).items())
    writer = SummaryWriter(log_dir='runs/'+augs.model_name)
    writer.add_text('Config Parameters', augs_str)

    if 'base' in augs.model_name:
        close_causal = True
        model = TrajCausalBase(name = augs.model_name,
                        c_in = augs.input_dim,
                        c_env = augs.env_dim,
                        c_out = output_dim,
                        embed_dim = augs.embed_dim,
                        n_filters = n_filters,
                        gru_layers = augs.gru_layers,
                        base_encoder = augs.base_encoder,
                        qk_dim = augs.latent_dim,
                        envbook_num = augs.envbook_num,
                        envbook_dim = augs.envbook_dim,
                        flag_meta = flag_meta,
                        cat_or_add = augs.cat_or_add,
                        with_random = augs.with_random,
                        without_point_attention = close_causal,
                        writer = writer
                        )
    else:
        close_causal = False
        model = TrajCausalFinal(name = augs.model_name,
                        c_in = augs.input_dim,
                        c_env = augs.env_dim,
                        c_out = output_dim,
                        embed_dim = augs.embed_dim,
                        n_filters = n_filters,
                        gru_layers = augs.gru_layers,
                        base_encoder = augs.base_encoder,
                        qk_dim = augs.latent_dim,
                        envbook_num = augs.envbook_num,
                        envbook_dim = augs.envbook_dim,
                        flag_meta = flag_meta,
                        cat_or_add = augs.cat_or_add,
                        with_random = augs.with_random,
                        without_point_attention = close_causal,
                        writer = writer
                        )

    print('=========param_num=========', model.param_num('none'))
    writer.add_text('Model param_num', str(model.param_num('none')))

    # model = torch.nn.DataParallel(model)

    trainer = CausalTrainer(model=model,
                      dataloader_dict=dataloader_dict,
                      base_lr=augs.base_lr,
                      steps=[35,70],
                      lr_decay_ratio=augs.lr_decay_ratio,
                      log_dir=augs.log_dir,
                      n_exp=augs.n_exp,
                      save_iter=augs.save_iter,
                      clip_grad_value=augs.max_grad_norm,
                      max_epochs=augs.max_epochs,
                      patience=augs.patience,
                      o = augs.o,
                      c = augs.c,
                      co = augs.co,
                      only_object = close_causal,
                      writer = writer
                      )

    if augs.mode == 'train':
        trainer.train()
        
        val_acc_c, val_acc_o, val_acc_co, test_acc_c, test_acc_o, test_acc_co = trainer.test(-1)
        text = f'val_acc_c: {val_acc_c}, val_acc_o: {val_acc_o}, val_acc_co: {val_acc_co}, ' + \
                f'test_acc_c: {test_acc_c}, test_acc_o: {test_acc_o}, test_acc_co: {test_acc_co}'
        writer.add_text('TestAcc', text)

        print(f"val_acc_o: {val_acc_o}")
        print(f"test_acc_o: {test_acc_o}")

        print(augs.model_name)
        
    else:
        trainer.test(-1)


if __name__ == "__main__":
    main()
