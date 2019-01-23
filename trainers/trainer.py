# encoding: utf-8
import math
import time
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.loss import euclidean_dist, hard_example_mining
from utils.meters import AverageMeter


class cls_tripletTrainer:
    def __init__(self, opt, model, optimzier, criterion, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer= optimzier
        self.criterion = criterion
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            self._parse_data(inputs)
            self._forward()
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('loss', self.loss.item(), global_step)
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))
        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        if self.opt.random_crop and random.random() > 0.3:
            h, w = imgs.size()[-2:]
            start = int((h-2*w)*random.random())
            mask = imgs.new_zeros(imgs.size())
            mask[:, :, start:start+2*w, :] = 1
            imgs = imgs * mask
        '''
        if random.random() > 0.5:
            h, w = imgs.size()[-2:]
            for attempt in range(100):
                area = h * w
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 3.33)
                ch = int(round(math.sqrt(target_area * aspect_ratio)))
                cw = int(round(math.sqrt(target_area / aspect_ratio)))
                if cw <  w and ch < h:
                    x1 = random.randint(0, h - ch)
                    y1 = random.randint(0, w - cw)
                    imgs[:, :, x1:x1+h, y1:y1+w] = 0
                    break
        '''
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, feat = self.model(self.data)
        self.loss = self.criterion(score, feat, self.target)

    def _backward(self):
        self.loss.backward()
