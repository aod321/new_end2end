import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from template import TemplateModel, F1Accuracy
from model import Stage2Model
from torch.utils.data import DataLoader
from dataset import PartsDataset, Stage2Augmentation
from prepgress import Stage2Resize, Stage2ToTensor
from torchvision import transforms
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import tensorboardX as tb
import uuid as uid
import os

uuid = str(uid.uuid1())[0:8]

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--cuda", default=0, type=int, help="Choose GPU with cuda number")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
parser.add_argument("--workers", default=4, type=int, help="workers ")

args = parser.parse_args()
print(args)

name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']


class TrainModel(TemplateModel):
    def __init__(self, dataset_class, txt_file, root_dir, transform, num_workers):
        super(TrainModel, self).__init__()
        self.writer = SummaryWriter('logs')
        self.args = args

        self.label_num = 6

        self.step = 0
        self.epoch = 0
        self.best_error = [float('Inf')
                           for _ in range(self.label_num)]
        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage2Model().to(self.device)
        self.optimizer = [optim.Adam(self.model.parameters(), self.args.lr)
                          for _ in range(self.label_num)]
        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()

        self.train_loader = None
        self.eval_loader = None

        self.ckpt_dir = "checkpoints_%s" % uuid
        self.scheduler = [optim.lr_scheduler.StepLR(self.optimizer[r], step_size=5, gamma=0.5)
                          for r in range(self.label_num)]
        self.load_dataset(dataset_class, txt_file, root_dir, transform, num_workers)

    def train_loss(self, batch):
        image = batch['image'].to(self.device)
        label = {x: batch['labels'][x].to(self.device)
                 for x in name_list
                 }

        # Image Shape(N, 6, 3, 64, 64)
        pred = self.model(image)
        loss = [self.criterion(pred[i], label[name_list[i]].argmax(dim=1, keepdim=False))
                for i in range(len(pred))]

        return loss

    def eval_error(self):
        # Image Shape(N, 6, 3, 64, 64)
        error = []
        counts = 0
        for i, batch in enumerate(self.eval_loader):
            counts += 1
            parts = batch['image'].to(self.device)
            labels = {x: batch['labels'][x].to(self.device)
                      for x in name_list
                      }
            pred = self.model(parts)
            for r in range(len(pred)):
                error.append((self.metric(pred[r],
                                          labels[name_list[r]].argmax(dim=1, keepdim=False)
                                          )).item()
                             )

        return error

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            error = self.eval_error()

        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)

        for j in range(6):
            if error[j] < self.best_error[j]:
                self.best_error[j] = error[j]
                self.save_state(os.path.join(self.ckpt_dir, 'best_%s.pth.tar' % name_list[j]), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)), False)
        self.writer.add_scalar('error_eyebrow1_%s' % uuid, error[0], self.epoch)
        self.writer.add_scalar('error_eyebrow2_%s' % uuid, error[1], self.epoch)
        self.writer.add_scalar('error_eye1_%s' % uuid, error[2], self.epoch)
        self.writer.add_scalar('error_eye2_%s' % uuid, error[3], self.epoch)
        self.writer.add_scalar('error_nose_%s' % uuid, error[4], self.epoch)
        self.writer.add_scalar('error_mouth_%s' % uuid, error[5], self.epoch)
        print('\n==============================')
        print('epoch {} finished\n'
              'error_eyebrow1 {:.3}\terror_eyebrow2 {:.3}\terror_eye1 {:.3}\t'
              'error_eye2 {:.3}\terror_nose {:.3}\terror_mouth {:.3}\n'
              'best_error_eyebrow1 {:.3}\tbest_error_eyebrow2 {:.3}\t'
              'best_error_eye1 {:.3}\tbest_error_eye2 {:.3}\tbest_error_nose {:.3}\tbest_error_mouth {:.3}\n'
              'best_error_mean {:.3}'
              .format(self.epoch, error[0], error[1], error[2], error[3], error[4], error[5],
                      self.best_error[0], self.best_error[1], self.best_error[2],
                      self.best_error[3], self.best_error[4], self.best_error[5],
                      np.mean(self.best_error)))
        print('==============================\n')
        if self.eval_logger:
            self.eval_logger(self.writer)

        torch.cuda.empty_cache()

    def train(self):
        self.model.train()
        self.epoch += 1
        for i, batch in enumerate(self.train_loader):
            self.step += 1
            for k in range(self.label_num):
                self.optimizer[k].zero_grad()
            loss = self.train_loss(batch)
            for k in range(self.label_num):
                loss[k].backward()
                self.optimizer[k].step()
            loss_item = [loss[k].item()
                         for k in range(self.label_num)]
            if self.step % args.display_freq == 0:
                self.writer.add_scalar('loss_eyebrow1_%s' % uuid, loss_item[0], self.step)
                self.writer.add_scalar('loss_eyebrow2_%s' % uuid, loss_item[1], self.step)
                self.writer.add_scalar('loss_eye1_%s' % uuid, loss_item[2], self.step)
                self.writer.add_scalar('loss_eye2_%s' % uuid, loss_item[3], self.step)
                self.writer.add_scalar('loss_nose_%s' % uuid, loss_item[4], self.step)
                self.writer.add_scalar('loss_mouth_%s' % uuid, loss_item[5], self.step)
                print('epoch {}\tstep {}\n'
                      'loss_eyebrow1 {:.3}\tloss_eyebrow2 {:.3}\t'
                      'loss_eye1 {:.3}\tloss_eye2 {:.3}\tloss_nose {:.3}\tloss_mouth {:.3}\n'
                      'loss_all_mean {:.3}'.format(
                    self.epoch, self.step, loss_item[0], loss_item[1], loss_item[2],
                    loss_item[3], loss_item[4], loss_item[5], np.mean(loss_item)
                ))
                if self.train_logger:
                    self.train_logger(self.writer)

        torch.cuda.empty_cache()

    def load_state(self, fname, optim=True, map_location=None):
        path = [os.path.join(fname, 'best_%s.pth.tar' % x)
                for x in name_list]
        state = [torch.load(path[i], map_location=map_location)
                 for i in range(self.label_num)]

        temp_state = []
        if isinstance(self.model, torch.nn.DataParallel):
            for i in range(5):
                self.model.module.load_state_dict(state[i]['model'])
                temp_state.append(self.model.module.single_model[i])
            temp_state.append(self.model.module.mouth_model)
            for i in range(5):
                self.model.module.single_model[i] = temp_state[i]
            self.model.module.mouth_model = temp_state[5]

        else:
            for i in range(5):
                self.model.load_state_dict(state[i]['model'])
                temp_state.append(self.model.single_model[i])
            temp_state.append(self.model.mouth_model)
            for i in range(5):
                self.model.single_model[i] = temp_state[i]
            self.model.mouth_model = temp_state[5]

            print('load model from {}'.format(fname))

    def load_dataset(self, dataset_class, txt_file, root_dir, transform, num_workers):

        data_after = Stage2Augmentation(dataset=dataset_class,
                                        txt_file=txt_file,
                                        root_dir=root_dir,
                                        resize=(64, 64)
                                        )

        Dataset = data_after.get_dataset()
        # Dataset = {x: dataset_class(txt_file=txt_file[x],
        #                             root_dir=root_dir,
        #                             transform=transform
        #                             )
        #            for x in ['train', 'val']
        #            }
        Loader = {x: DataLoader(Dataset[x], batch_size=args.batch_size,
                                shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']
                  }
        self.train_loader = Loader['train']
        self.eval_loader = Loader['val']

        return Loader


def start_train(model_path=None):
    dataset_class = PartsDataset
    txt_file_names = {
        'train': "exemplars.txt",
        'val': "tuning.txt"
    }
    root_dir = "/data1/yinzi/facial_parts"
    transform = transforms.Compose([Stage2Resize((64, 64)),
                                    Stage2ToTensor()
                                    ])

    train = TrainModel(dataset_class, txt_file_names, root_dir, transform, num_workers=args.workers)
    if model_path:
        train.load_state(model_path)

    for epoch in range(args.epochs):
        train.train()
        for i in range(6):
            train.scheduler[i].step(epoch)
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
# start_train(model_path="/home/yinzi/data3/vim
