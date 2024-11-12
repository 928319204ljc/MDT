#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import time
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 创建文件夹用于保存图像
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)



class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/EGait_journal/train_diff_combine_double_score_fagg.yaml',
        # default='./config/kinetics-skeleton/train_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 2],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=20,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument('--train_ratio', default=0.9)
    parser.add_argument('--val_ratio', default=0.0)
    parser.add_argument('--test_ratio', default=0.1)

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=2, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=2, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--save_model', default=False)
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.best_val_accuracies = []  # 用于存储每个 epoch 的最高测试集准确率
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.ap_values = []
        self.map_values = []
        # 添加新的属性列表来存储训练和验证的 mAP 值
        self.train_map_values = []  # 用于存储训练集的 mAP 值
        self.val_map_values = []    # 用于存储验证集的 mAP 值
        self.save_path = os.path.join(self.arg.work_dir, 'training_plots')
        create_folder(self.save_path)
        if arg.phase == 'train':
            self.save_arg()
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        # train_args, test_args = self.arg.train_feeder_args, self.arg.test_feeder_args
        # train_ratio, val_ratio, test_ratio = self.arg.train_ratio, self.arg.val_ratio, self.arg.test_ratio
        # my_feeder = FeederSplit(train_data_m_path=train_args['data_m_path'], train_data_p_path=train_args['data_p_path'],
        #                                 train_label_path=train_args['label_path'], train_feature_path=train_args['feature_path'],
        #                                 test_data_m_path=test_args['data_m_path'], test_data_p_path=test_args['data_p_path'],
        #                                 test_label_path=test_args['label_path'], test_feature_path=test_args['feature_path'],
        #                                 train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        # train_set, test_set = my_feeder.get_data()
        # self.data_loader['train'] = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.arg.batch_size, shuffle=True, num_workers=self.arg.num_worker, drop_last=True, worker_init_fn=init_seed)
        # self.data_loader['test'] = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.arg.test_batch_size, shuffle=False, num_workers=self.arg.num_worker, drop_last=False,worker_init_fn=init_seed)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):

        output_device = 0
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        # print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        # self.loss = nn.BCELoss().cuda(output_device)
        self.loss2 = nn.MSELoss().cuda(output_device)
        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False

        train_class_total_num = np.array([0, 0, 0, 0])
        train_class_true_num = np.array([0, 0, 0, 0])
        total_acc, cnt = 0, 0
        all_labels = []
        all_outputs = []
        loss_value = []
        accuracy_value = []
        for batch_idx, (data_m, data_p, label, feature, index) in enumerate(process):
            train_mode = 'MLL' if len(label.size()) > 1 else 'SLL'
            self.global_step += 1

            # 获取数据
            data_m = Variable(data_m.float().cuda(self.output_device), requires_grad=False)
            data_p = Variable(data_p.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            if train_mode == 'MLL':
                label = label.to(torch.float32)
            feature = Variable(feature.float().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # 前向传播
            output_p, output2, output_m = self.model(data_p, data_m)
            output = (output_m + output_p) / 2

            if train_mode == 'MLL':
                output_p = F.sigmoid(output_p)
                output_m = F.sigmoid(output_m)

            loss1_m = self.loss(output_m, label)
            loss1_p = self.loss(output_p, label)
            loss2 = self.loss2(output2, feature)
            loss = loss1_m + loss1_p + loss2

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            # 收集所有标签和输出以计算AP和mAP
            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(output.detach().cpu().numpy())

            if train_mode == 'SLL':
                value, predict_label = torch.max(output.data, 1)
                total_acc += torch.sum((predict_label == label.data).float())
                cnt += label.size(0)
                trues = list(label.data.cpu().numpy())
                for idx, lb in enumerate(predict_label):
                    train_class_total_num[trues[idx]] += 1
                    train_class_true_num[trues[idx]] += int(lb == trues[idx])
            else:
                total_acc += torch.round(output).eq(label).sum()
                cnt += label.numel()
                class_total_num = torch.round(output).eq(1).sum(axis=0)
                class_true_num = (torch.round(output).eq(label) & label.eq(1)).sum(axis=0)
                for idx in range(len(class_total_num)):
                    train_class_total_num[idx] += class_total_num[idx]
                for idx in range(len(class_true_num)):
                    train_class_true_num[idx] += class_true_num[idx]

            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_1m', loss1_m, self.global_step)
            self.train_writer.add_scalar('loss_1p', loss1_p, self.global_step)
            self.train_writer.add_scalar('loss_2', loss2, self.global_step)

            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            loss_value.append(loss.data.item())
            accuracy_value.append((torch.max(output.data, 1)[1] == label.data).float().mean().item())

            # 记录损失和准确率
        self.train_losses.append(np.mean(loss_value))
        self.train_accuracies.append(np.mean(accuracy_value))

        # 计算AP和加权mAP
        if len(all_labels) > 0:
            all_labels = np.array(all_labels)
            all_outputs = np.array(all_outputs)

            # 确保是二维数组
            if len(all_labels.shape) == 1:
                all_labels = all_labels.reshape(-1, 1)
            if len(all_outputs.shape) == 1:
                all_outputs = all_outputs.reshape(-1, 1)

            # 将标签转换为 one-hot 编码格式
            all_labels_one_hot = np.eye(4)[all_labels.astype(int).flatten()]

            ap = []
            for i in range(all_labels_one_hot.shape[1]):  # 遍历每个类别
                ap_i = average_precision_score(all_labels_one_hot[:, i], all_outputs[:, i])
                ap.append(ap_i)

            # 设定类别权重
            weights = np.array([1.0, 1.0, 1.0, 1.0])  # 假设按照 'Angry', 'Happy', 'Sad', 'Neutral' 的顺序

            # 计算加权mAP
            weighted_mAP = np.average(ap, weights=weights)

            self.print_log('AP per class: {}'.format(ap))
            self.print_log('Weighted mAP: {:.4f}'.format(weighted_mAP))
            self.train_writer.add_scalar('Weighted mAP', weighted_mAP, self.global_step)

            # 将训练集的mAP值存储到train_map_values列表中
            self.train_map_values.append(weighted_mAP)

        # 保留原有的损失和时间统计输出
        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values())))) for k, v in timer.items()}
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        print('Happy:{},Sad:{},Angry:{},Neutral:{}'.format(train_class_true_num[0] * 1.0 / train_class_total_num[0],
                                                           train_class_true_num[1] * 1.0 / train_class_total_num[1],
                                                           train_class_true_num[2] * 1.0 / train_class_total_num[2],
                                                           train_class_true_num[3] * 1.0 / train_class_total_num[3]))
        print('Train Accuracy: {: .2f}%'.format(100 * total_acc * 1.0 / cnt))

        # 生成训练集混淆矩阵
        if epoch == self.arg.num_epoch - 1:  # 仅在最后一个epoch生成
            cm_train = confusion_matrix(all_labels, np.argmax(all_outputs, axis=1))
            disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train,
                                                display_labels=['Happy', 'Sad', 'Angry', 'Neutral'])
            disp_train.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix - Training Set')
            plt.savefig(os.path.join(self.save_path, 'confusion_matrix_train.png'))
            plt.close()
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        self.model.eval()
        loss_value = []
        accuracy_value = []
        test_class_total_num = np.array([0, 0, 0, 0])
        test_class_true_num = np.array([0, 0, 0, 0])
        total_acc, cnt = 0, 0
        all_labels = []
        all_outputs = []

        self.print_log('Eval epoch: {}'.format(epoch + 1))

        for ln in loader_name:
            loss_value = []
            process = tqdm(self.data_loader[ln])

            for batch_idx, (data_m, data_p, label, feature, index) in enumerate(process):
                test_mode = 'MLL' if len(label.size()) > 1 else 'SLL'
                with torch.no_grad():
                    data_m = Variable(data_m.float().cuda(self.output_device), requires_grad=False)
                    data_p = Variable(data_p.float().cuda(self.output_device), requires_grad=False)
                    label = Variable(label.long().cuda(self.output_device), requires_grad=False)

                    if test_mode == 'MLL':
                        label = label.to(torch.float32)

                    # Forward pass
                    output_p, output2, output_m = self.model(data_p, data_m)
                    output = (output_m + output_p) / 2

                    if test_mode == 'MLL':
                        output = F.sigmoid(output)

                    loss = self.loss(output, label)
                    loss_value.append(loss.data.item())

                    # Collect outputs and labels for AP/mAP calculation
                    all_labels.extend(label.cpu().numpy())
                    all_outputs.extend(output.cpu().numpy())

                    # Calculate accuracy
                    _, predict_label = torch.max(output.data, 1)
                    correct = (predict_label == label).sum().item()
                    total_acc += correct
                    cnt += label.size(0)

                    # Update class-wise accuracy
                    trues = list(label.data.cpu().numpy())
                    for idx, lb in enumerate(predict_label):
                        test_class_total_num[trues[idx]] += 1
                        test_class_true_num[trues[idx]] += int(lb == trues[idx])
                    loss_value.append(loss.data.item())
                    accuracy_value.append((torch.max(output.data, 1)[1] == label.data).float().mean().item())

            # 记录损失和准确率
            self.val_losses.append(np.mean(loss_value))
            self.val_accuracies.append(np.mean(accuracy_value))

            # Calculate AP and mAP
            if len(all_labels) > 0:
                # 确保为weights赋值
                weights = np.array([1.0, 1.0, 1.0, 1.0])  # 假设按照 'Angry', 'Happy', 'Sad', 'Neutral' 的顺序
                all_labels = np.array(all_labels)
                all_outputs = np.array(all_outputs)

                # 确保是二维数组
                if len(all_labels.shape) == 1:
                    all_labels = all_labels.reshape(-1, 1)
                if len(all_outputs.shape) == 1:
                    all_outputs = all_outputs.reshape(-1, 1)

                # 将标签转换为 one-hot 编码格式
                all_labels_one_hot = np.eye(4)[all_labels.astype(int).flatten()]

                ap = []
                for i in range(all_labels_one_hot.shape[1]):  # 遍历每个类别
                    ap_i = average_precision_score(all_labels_one_hot[:, i], all_outputs[:, i])
                    ap.append(ap_i)

                # 计算加权mAP
                weighted_mAP = np.average(ap, weights=weights)

                self.ap_values.append(ap)
                self.val_map_values.append(weighted_mAP)

                self.print_log('AP per class: {}'.format(ap))
                self.print_log('Weighted mAP: {:.4f}'.format(weighted_mAP))
                self.val_writer.add_scalar('Weighted mAP', weighted_mAP, self.global_step)

            # Calculate overall accuracy
            accuracy = total_acc / cnt
            self.print_log('Current epoch accuracy: {:.2f}%'.format(accuracy * 100))

            # Update best accuracy
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.print_log('Best accuracy updated: {:.2f}%'.format(self.best_acc * 100))
            else:
                self.print_log('No improvement in best accuracy.')

            # 添加用于记录最高测试集准确率的代码
            self.best_val_accuracies.append(self.best_acc)

            # Log final statistics
            self.print_log(
                '\tMean {} loss of {} batches: {}.'.format(ln, len(self.data_loader[ln]), np.mean(loss_value)))
            print('Top1: {:.2f}%'.format(accuracy * 100))
            self.print_log('Best acc: {:.2f}%'.format(self.best_acc * 100))
            print('Happy:{},Sad:{},Angry:{},Neutral:{}'.format(
                test_class_true_num[0] * 1.0 / test_class_total_num[0],
                test_class_true_num[1] * 1.0 / test_class_total_num[1],
                test_class_true_num[2] * 1.0 / test_class_total_num[2],
                test_class_true_num[3] * 1.0 / test_class_total_num[3]
            ))

        # 生成测试集混淆矩阵
        if epoch == self.arg.num_epoch - 1:  # 仅在最后一个epoch生成
            cm_test = confusion_matrix(all_labels, np.argmax(all_outputs, axis=1))
            disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                                   display_labels=['Happy', 'Sad', 'Angry', 'Neutral'])
            disp_test.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix - Test Set')
            plt.savefig(os.path.join(self.save_path, 'confusion_matrix_test.png'))
            plt.close()

            # Optionally save score
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump({'labels': all_labels, 'outputs': all_outputs}, f)

        if wrong_file is not None:
            f_w.close()
        if result_file is not None:
            f_r.close()

        self.print_log('Done.\n')

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-6:
                    break
                # save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                #         epoch + 1 == self.arg.num_epoch)
                save_model = False
                start = time.time()
                self.train(epoch, save_model=save_model)
                end = time.time()
                print(end - start)

                start = time.time()
                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])
                end = time.time()
                print(end - start)

            # print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)
            self.print_log('best accuracy: {}'.format(self.best_acc))
            self.plot_and_save()
        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '/wrong.txt'
                rf = self.arg.model_saved_name + '/right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

    def plot_and_save(self):
        epochs = range(1, len(self.train_losses) + 1)

        # 绘制损失曲线
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Training loss')
        plt.plot(epochs, self.val_losses, label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'loss_curve.png'))

        # 绘制准确率曲线
        plt.figure()
        plt.plot(epochs, self.train_accuracies, label='Training accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'accuracy_curve.png'))


        # 绘制测试集最高准确率曲线
        if len(self.best_val_accuracies) > 0:
            plt.figure()
            plt.plot(epochs, self.best_val_accuracies, label='Validation Accuracy')
            plt.title('Validation Accuracy over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(self.save_path, 'best_val_accuracy_curve.png'))

        # 绘制mAP曲线
        if len(self.train_map_values) > 0 and len(self.val_map_values) > 0:
            plt.figure()
            plt.plot(epochs, self.train_map_values, label='Training mAP')
            plt.plot(epochs, self.val_map_values, label='Validation mAP')
            plt.title('mAP over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('mAP')
            plt.legend()
            plt.savefig(os.path.join(self.save_path, 'map_curve.png'))


        # 绘制AP曲线
        if len(self.ap_values) > 0:
            plt.figure()
            # 定义类别名称
            class_names = ['Happy', 'Sad', 'Angry', 'Neutral']
            for i, ap_class in enumerate(zip(*self.ap_values)):
                plt.plot(epochs, ap_class, label=f'AP {class_names[i]}')
            plt.title('AP per Class over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('AP')
            plt.legend()
            plt.savefig(os.path.join(self.save_path, 'ap_curve.png'))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
