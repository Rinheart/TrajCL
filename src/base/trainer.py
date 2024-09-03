import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.utils.rnn as rnn_utils

from utils.logger import get_logger

class BaseTrainer():
    def __init__(
            self,
            model: nn.Module,
            dataloader_dict,
            base_lr: float,
            steps,
            lr_decay_ratio,
            log_dir: str,
            n_exp: int,
            save_iter: int = 300,
            clip_grad_value: Optional[float] = None,
            max_epochs: Optional[int] = 1000,
            patience: Optional[int] = 1000,
            device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()

        self._logger = get_logger(
            log_dir, __name__, 'info_{}.log'.format(n_exp), level=logging.INFO)
        if device is None:
            print("`device` is missing, try to train and evaluate the model on default device.")
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)

        self._model = model.to(self._device)
        # self._logger.info("the number of parameters: {}".format(self.model.param_num(self.model.name))) 

        self._loss_fn = nn.CrossEntropyLoss()
        self._loss_fn.to(self._device)
        self._base_lr = base_lr
        self._optimizer = Adam(self.model.parameters(), base_lr)

        if lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(self.optimizer,
                                             steps,
                                             gamma=lr_decay_ratio)
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_iter = save_iter
        self._save_path = log_dir
        self._n_exp = n_exp
        self._dataloader_dict = dataloader_dict

    @property
    def model(self):
        return self._model


    @property
    def dataloader_dict(self):
        return self._dataloader_dict
    
    @property
    def data(self):
        return self._dataloader_dict

    @property
    def logger(self):
        return self._logger

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def device(self):
        return self._device

    @property
    def save_path(self):
        return self._save_path

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # filename = 'epoch_{}.pt'.format(epoch)
        filename = 'final_model_{}.pt'.format(n_exp)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
        return filename
        # return True

    def load_model(self, epoch, save_path, n_exp):
        filename = 'final_model_{}.pt'.format(n_exp)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))
        return True

    def early_stop(self, epoch, best_acc):
        self.logger.info('Early stop at epoch {}, val accuracy = {:.6f}'.format(epoch, best_acc))

    def train_batch(self, batch_data, iter):
        # print(iter)
        X, M, Y = self._check_device([batch_data[0], batch_data[2], batch_data[-1]])
        X_ = rnn_utils.pack_padded_sequence(X, batch_data[1], batch_first=True)
        self.optimizer.zero_grad()
        pred = self.model(X_, M)
        loss = self.loss_fn(pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.logger.info("start training !!!!!")

        # training phase
        iter = 0
        val_accs = [-1]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, max(val_accs))
                break

            start_time = time.time()
            for i, batch_data in enumerate(self.dataloader_dict['train']):
                # print(iter)
                # st = time.time()
                train_losses.append(self.train_batch(batch_data, iter))
                # print(time.time() - st)
                iter += 1
            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_acc = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_lr()[0]

            message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_acc: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch,
                                 self._max_epochs,
                                 iter,
                                 np.mean(train_losses),
                                 val_acc,
                                 new_lr,
                                 (end_time - start_time))
            self._logger.info(message)

            if val_acc > np.max(val_accs):
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val accuracy increases from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.max(val_accs), val_acc, model_file_name))
                val_accs.append(val_acc)
                saved_epoch = epoch

    def test_batch(self, batch_data):
        X, M = self._check_device([batch_data[0], batch_data[2]])
        X_ = rnn_utils.pack_padded_sequence(X, batch_data[1], batch_first=True)
        pred = self.model(X_, M)
        return pred

    def evaluate(self, mode='valid'):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, batch_data in enumerate(self.dataloader_dict[mode]):
                pred = self.test_batch(batch_data)
                labels.append(batch_data[-1])
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        _, outputs = torch.max(preds, 1)
        acc = (outputs == labels).sum() * 1.0 / len(labels)
        return acc

    def test(self, epoch):
        self.load_model(epoch, self.save_path, self._n_exp)

        val_acc = self.evaluate('valid')
        test_acc = self.evaluate('test')
        print('val_acc: {:.6f}, test_acc: {:.6f}'.format(val_acc, test_acc))

        filename = os.path.join(self.save_path, 'test_results_{}.csv'.format(self._n_exp))
        # save to csv
        np.savetxt(filename, [val_acc, test_acc], fmt='%.6f', delimiter=',')
        return val_acc, test_acc

    def save_preds(self, epoch):
        pass