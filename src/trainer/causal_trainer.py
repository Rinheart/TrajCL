import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import torch
import pickle
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from utils.logger import get_logger
from base.trainer import BaseTrainer


class CausalTrainer(BaseTrainer):
    def __init__(self, o, c, co, only_object, writer, **args):
        super(CausalTrainer, self).__init__(**args)
        
        self.o = o
        self.c = c
        self.co = co
        self.only_object = only_object
        
        self.gamma = 10
        self.theta = 5e-5

        self.writer = writer
        self.tb_graph = 0

    def train_batch(self, batch_data, iter):
        X, M, Y = self._check_device([batch_data[0], batch_data[2], batch_data[-1]])
        _len = torch.tensor(batch_data[1])

        self.optimizer.zero_grad()
        out_c_logis, out_o_logis, out_co = self.model(X, M, _len)
        
        if self.tb_graph:
            self.writer.add_graph(self.model, (X, M, _len))
            self.tb_graph=0

        # loss calculation
        if out_c_logis is out_o_logis is None:
            loss = self.loss_fn(out_co, Y)
            c_loss = torch.full((1,), float('nan'))
            o_loss = torch.full((1,), float('nan'))
            co_loss = torch.full((1,), float('nan'))
        else:
            uniform_target = torch.ones_like(out_c_logis, dtype=torch.float).to(self._device) / self.model.c_out
 
            c_loss = F.kl_div(out_c_logis, uniform_target, reduction='batchmean')
            o_loss = F.nll_loss(out_o_logis, Y)
            co_loss = self.loss_fn(out_co, Y)

            if self.only_object:
                loss = o_loss
            else:
                loss = self.c * c_loss + self.o * o_loss + self.co * co_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()

        return loss.item(), c_loss.item(), o_loss.item(), co_loss.item()
    

    def train(self):
        self.logger.info("start training !!!!!")

        # training phase
        iter = 0
        val_accs = [-1]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            c_losses = []
            o_losses = []
            co_losses = []

            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, max(val_accs))
                break

            start_time = time.time()
            for i, batch_data in enumerate(self.dataloader_dict['train']):
                # st = time.time()
                
                loss, c_loss, o_loss, co_loss = self.train_batch(batch_data, iter)
                train_losses.append(loss)
                c_losses.append(c_loss)
                o_losses.append(o_loss)
                co_losses.append(co_loss)

                # print('Epoch: {}, Iter: {}, Batch: {}, Time: {:.2f}'.format(epoch, iter, i, time.time() - st))
                iter += 1
            end_time = time.time()
            self.logger.info("epoch complete, evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            acc_c, acc_o, acc_co = self.evaluate()
            val_acc = acc_o

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]
            
            ###############################logs#########################################
            
            if self.model.point_att!=None:
                poatt_c = self.model.point_att[:,:,0].unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)  # 1, 1, 256, 100 NCHW
                poatt_o = self.model.point_att[:,:,1].unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)
                poatt_c = poatt_c.repeat_interleave(6, dim=-1).repeat_interleave(6, dim=-2)
                poatt_o = poatt_o.repeat_interleave(6, dim=-1).repeat_interleave(6, dim=-2)
                distribut_co = self.model.point_att.flatten()

                self.writer.add_images(img_tensor=poatt_c, tag='context_PoAtt_onebatch', global_step=epoch)
                self.writer.add_images(img_tensor=poatt_o, tag='object_PoAtt_onebatch', global_step=epoch)
                self.writer.add_histogram('PoAtt_COall_distri', distribut_co, epoch, bins=100)
                self.writer.add_histogram('PoAtt_context', poatt_c, epoch)
                self.writer.add_histogram('PoAtt_object', poatt_o, epoch)

            if self.model.env_selects!=None:
                plotselects = self.model.env_selects[0].unsqueeze(0).unsqueeze(0).repeat_interleave(6, dim=-1).repeat_interleave(6, dim=-2)

                self.writer.add_images(img_tensor=plotselects, tag='env_selects', global_step=epoch)


            acc_c_train, acc_o_train, acc_co_train = self.evaluate(mode='train')
            self.writer.add_scalar("C_acc_train/train_acc", acc_c_train, epoch)
            self.writer.add_scalar("O_acc_train/train_acc", acc_o_train, epoch)
            self.writer.add_scalar("CO_acc_train/train_acc", acc_co_train, epoch)

            self.writer.add_scalar("All_Loss/train", np.mean(train_losses), epoch)
            self.writer.add_scalar("C_Loss/train", np.mean(c_losses), epoch)
            self.writer.add_scalar("O_Loss/train", np.mean(o_losses), epoch)
            self.writer.add_scalar("CO_Loss/train", np.mean(co_losses), epoch)

            self.writer.add_scalar("C_acc/val", acc_c, epoch)
            self.writer.add_scalar("O_acc(final acc)/val", acc_o, epoch)
            self.writer.add_scalar("CO_acc/val", acc_co, epoch)
            
            self.writer.add_scalar("LearningRate", new_lr, epoch)

            loss_message = 'Epoch [{}/{}] ({}) train_loss: {:.4f} / c:{:.4f} / o:{:.4f} / co:{:.4f}, lr: {:.6f}, {:.1f}s'.format(
                epoch,
                self._max_epochs,
                iter,
                np.mean(train_losses),
                np.mean(c_losses),
                np.mean(o_losses),
                np.mean(co_losses),
                new_lr,
                (end_time - start_time))

            val_message = 'Epoch [{}/{}] ({}) val_acc: c:{:.4f} / o:{:.4f} / co:{:.4f}'.format(
                epoch,
                self._max_epochs,
                iter,
                acc_c,
                acc_o,
                acc_co)

            self._logger.info(loss_message)
            self._logger.info(val_message)
            ########################################################################

            if val_acc > np.max(val_accs):
                model_file_name = self.save_model(epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val accuracy increases from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.max(val_accs), val_acc, model_file_name))
                val_accs.append(val_acc)
                saved_epoch = epoch

        self.logger.info('Save model at epoch {}, file at {}'.format(saved_epoch, self._save_path+model_file_name))


    def test_batch(self, batch_data):
        X, M = self._check_device([batch_data[0], batch_data[2]])
        _len = torch.tensor(batch_data[1])
        # X_ = rnn_utils.pack_padded_sequence(X, batch_data[1], batch_first=True)
        out_c_logis, out_o_logis, out_co = self.model(X, M, _len)
        return out_c_logis, out_o_logis, out_co


    def evaluate(self, mode='valid'):
        labels = []
        preds_c = []
        preds_o = []
        preds_co = []
        with torch.no_grad():
            self.model.eval()
            for _, batch_data in enumerate(self.dataloader_dict[mode]):
                out_c_logis, out_o_logis, out_co = self.test_batch(batch_data)
                labels.append(batch_data[-1])
                preds_c.append(out_c_logis.cpu())
                preds_o.append(out_o_logis.cpu())
                preds_co.append(out_co.cpu())

        labels = torch.cat(labels, dim=0)
        preds_c = torch.cat(preds_c, dim=0)
        preds_o = torch.cat(preds_o, dim=0)
        preds_co = torch.cat(preds_co, dim=0)

        _, outputs_c = torch.max(preds_c, 1)
        _, outputs_o = torch.max(preds_o, 1)
        _, outputs_co = torch.max(preds_co, 1)

        acc_c = (outputs_c == labels).sum() * 1.0 / len(labels)
        acc_o = (outputs_o == labels).sum() * 1.0 / len(labels)
        acc_co = (outputs_co == labels).sum() * 1.0 / len(labels)

        return acc_c, acc_o, acc_co    
    

    def test(self, epoch):
        self.load_model(epoch, self.save_path, self._n_exp)

        val_acc_c, val_acc_o, val_acc_co = self.evaluate('valid')
        test_acc_c, test_acc_o, test_acc_co = self.evaluate('test')

        print('val_acc: c:{:.6f} / o:{:.6f} / co:{:.6f}'.format(val_acc_c, val_acc_o, val_acc_co))
        print('test_acc: c:{:.6f} / o:{:.6f} / co:{:.6f}'.format(test_acc_c, test_acc_o, test_acc_co))

        filename = os.path.join(self.save_path, 'test_results_{}.csv'.format(self._n_exp))

        # save to csv
        results = np.array([[val_acc_c, val_acc_o, val_acc_co], [test_acc_c, test_acc_o, test_acc_co]])
        np.savetxt(filename, results, fmt='%.6f', delimiter=',')

        return val_acc_c, val_acc_o, val_acc_co, test_acc_c, test_acc_o, test_acc_co
    