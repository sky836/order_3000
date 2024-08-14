import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from models import PatchTST
from utils.metrics import masked_mae


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'transformer': PatchTST
        }
        model = model_dict[self.args.model].Model(self.args).float()
        print('###### generator parameters:', sum(param.numel() for param in model.parameters()))

        # nn.DataParallel允许在多个GPU上并行地训练模型
        # device_ids指定了要使用的GPU设备的ID列表。这意味着模型将被复制到指定的GPU设备上，
        # 每个GPU设备都会处理一部分数据，然后梯度将被汇总并用于更新模型参数。
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(args=self.args, flag=flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        return optimizer

    def _select_criterion(self):
        criterion = masked_mae
        return criterion

    def train(self, settings):
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='valid')
        test_data, test_loader = self._get_data(flag='test')

        criterion = self._select_criterion()
        optimizer = self._select_optimizer()

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(
            summary(
                self.model,
                [
                    (self.args.batch_size, self.args.seq_len, self.args.enc_in)
                ],
                verbose=0,  # avoid print twice
            )
        )
        print('number of params (M): %.2f' % (n_parameters / 1e6))

        # path = os.path.join(self.args.checkpoints, settings)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        path = '/kaggle/working/'

        # tensorboard_path = os.path.join('./runs/{}/'.format(settings))
        # if not os.path.exists(tensorboard_path):
        #     os.makedirs(tensorboard_path)
        tensorboard_path = '/kaggle/working/'

        writer = SummaryWriter(log_dir=tensorboard_path)

        step, n_epochs, best_loss, early_stop_count = 0, self.args.n_epochs, math.inf, 0
        train_steps = len(train_loader)
        time_now = time.time()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='min', factor=0.1, patience=5,
                                                                  verbose=False, threshold=0.001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=1e-7, eps=1e-08)

        for epoch in range(n_epochs):
            iter_count = 0
            self.model.train()  # 将模型设置为训练模式
            loss_record = []  # 记录每次epoch的损失值

            for i, (x, y, _, _) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()  # 将梯度清零
                x, y = x.float().to('cuda'), y.float().to('cuda')
                pred = self.model(x)

                batch_size, pred_len, n_channels = pred.shape
                pred = train_data.inverse_transform(pred.reshape(-1, n_channels)).reshape(batch_size, pred_len, n_channels)

                loss = criterion(y, pred)
                loss.backward()  # 计算梯度，误差逆传播算法
                optimizer.step()  # 更新参数
                step += 1
                loss_record.append(loss)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((n_epochs - epoch) * train_steps - i)
                    print("\tepoch: {1} | iters: {0} | loss: {2:.7f} | speed: {3:.4f}s/iter | left time: {4:.4f}s".
                              format(i + 1, epoch + 1, loss.item(), speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            mean_train_loss = sum(loss_record) / len(loss_record)
            writer.add_scalar(scalar_value=mean_train_loss, global_step=step, tag='Loss/train')

            # 模型验证
            mean_valid_loss = self.valid(valid_loader, criterion, valid_data)
            print(f'\nEpoch[{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss: .4f}, Valid loss: {mean_valid_loss: .4f}')
            lr_scheduler.step(mean_valid_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print("Epoch: {} current lr: {}".format(epoch + 1, current_lr))
            writer.add_scalar(scalar_value=mean_valid_loss, global_step=step, tag='Loss/valid')

            if mean_valid_loss < best_loss:
                best_loss = mean_valid_loss
                best_model_path = path + '/' + 'checkpoint.pth'
                torch.save(self.model.state_dict(), os.path.join(best_model_path))
                print('Saving model with loss {:.3f}...'.format(best_loss))
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.args.patience:
                print('\n模型效果没有提升，所以终止训练！')
                break

    def valid(self, valid_loader, criterion, valid_data):
        # 模型验证
        self.model.eval()
        loss_record = []
        for x, y, _, _ in valid_loader:
            x, y = x.float().to('cuda'), y.float().to('cuda')
            with torch.no_grad():
                pred = self.model(x)

                batch_size, pred_len, n_channels = pred.shape
                pred = valid_data.inverse_transform(pred.reshape(-1, n_channels)).reshape(batch_size, pred_len,
                                                                                          n_channels)
                loss = criterion(pred, y)
            loss_record.append(loss)

        mean_valid_loss = sum(loss_record) / len(loss_record)
        return mean_valid_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            print("Model = %s" % str(self.model))
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('number of params (M): %.2f' % (n_parameters / 1.e6))
        loss_record = []
        preds = []
        trues = []
        for x, y, _, _ in test_loader:
            x, y = x.float().to('cuda'), y.float().to('cuda')
            with torch.no_grad():
                pred = self.model(x)

                batch_size, pred_len, n_channels = pred.shape
                pred = test_data.inverse_transform(pred.reshape(-1, n_channels)).reshape(batch_size, pred_len,
                                                                                          n_channels)
                loss = masked_mae(pred, y)

                preds.append(pred.detach().cpu().numpy().astype(np.float32))
                trues.append(y.detach().cpu().numpy().astype(np.float32))

            loss_record.append(loss)
        mean_test_loss = sum(loss_record) / len(loss_record)
        print('test_loss:', mean_test_loss)

        preds = np.array(preds)
        preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
        trues = np.array(trues)
        trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

