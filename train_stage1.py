import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from template import TemplateModel, F1Accuracy
from model import Stage1Model
from torch.utils.data import DataLoader
from dataset import HelenDataset
from prepgress import Resize, ToPILImage, ToTensor
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
args = parser.parse_args()
print(args)

# Dataset Read_in Part
root_dir = "/data1/yinzi/datas"
# root_dir = '/home/yinzi/Downloads/datas'

txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

transforms_list = {
    'train':
        transforms.Compose([
            ToPILImage(),
            # RandomRotation(15),
            # RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
            # CenterCrop((512,512)),
            Resize((64, 64)),
            ToTensor()
            # Normalize()
        ]),
    'val':
        transforms.Compose([
            ToPILImage(),
            Resize((64, 64)),
            ToTensor()
            # Normalize()
        ])
}
# Stage 1 augmentation
stage1_dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                                           root_dir=root_dir,
                                           transform=transforms.Compose([
                                               ToPILImage(),
                                               Resize((64, 64)),
                                               ToTensor()
                                           ])
                                           )
                            for x in ['train', 'val']
                            }
stage1_dataloaders = {x: DataLoader(stage1_dataset[x], batch_size=args.batch_size,
                                    shuffle=True, num_workers=4)
                      for x in ['train', 'val']}

stage1_dataset_sizes = {x: len(stage1_dataset[x]) for x in ['train', 'val']}


class TrainModel(TemplateModel):

    def __init__(self, argus=args):
        super(TrainModel, self).__init__()
        self.label_channels = 9
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')
        self.best_accu = float('-Inf')

        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

        self.model = Stage1Model().to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.CrossEntropyLoss()
        # self.metric = F1Accuracy()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = stage1_dataloaders['train']
        self.eval_loader = stage1_dataloaders['val']

        self.ckpt_dir = "checkpoints_%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        image, label = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)

        pred, _, _ = self.model(image, label)
        # loss = self.criterion(pred, y)
        loss = self.criterion(pred, label.long())
        # loss /= self.args.batch_size
        # gt_grid = torchvision.utils.make_grid(torch.unsqueeze(label, dim=1))
        # predicts = pred.argmax(dim=1, keepdim=False)
        # predicts = torch.unsqueeze(predicts, dim=1)
        # predicts_grid = torchvision.utils.make_grid(predicts)
        # self.writer.add_image('ground_truth_%s' % uuid, gt_grid[0], global_step=self.step, dataformats='HW')
        # self.writer.add_image('predicts_%s' % uuid, predicts_grid[0], global_step=self.step, dataformats='HW')
        return loss, None

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            pred, _, _ = self.model(image, label)
            error = self.metric(pred, label.long())
            # error = self.criterion(pred, y)

            loss_list.append(error.item())

        return np.mean(loss_list), None

    def train(self):
        self.model.train()
        self.epoch += 1
        for batch in self.train_loader:
            self.step += 1
            self.optimizer.zero_grad()

            loss, others = self.train_loss(batch)

            loss.backward()
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_%s' % uuid, loss.item(), self.step)

                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))
                if self.train_logger:
                    self.train_logger(self.writer, others)


class Train_F1_eval(TrainModel):

    def eval(self):
        self.model.eval()
        accu, others = self.eval_accu()

        if accu > self.best_accu:
            self.best_accu = accu
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('accu_%s' % uuid, accu, self.epoch)
        print('epoch {}\taccu {:.3}\tbest_accu {:.3}'.format(self.epoch, accu, self.best_accu))

        if self.eval_logger:
            self.eval_logger(self.writer, others)

        return accu

    def eval_accu(self):
        accu_list = []
        # pred_list = []
        # label_list = []
        for batch in self.eval_loader:
            image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
            pred, _, _ = self.model(image, label)
            # pred_list.append(pred)
            # label_list.append(label)
            accu = self.metric(pred, label.long())
            accu_list.append(accu)
        # preds = torch.cat(pred_list, dim=0)
        # labels = torch.cat(label_list, dim=0)
        # imshow
        gt_grid = torchvision.utils.make_grid(torch.unsqueeze(label, dim=1))
        predicts = pred.argmax(dim=1, keepdim=False)
        predicts = torch.unsqueeze(predicts, dim=1)
        predicts_grid = torchvision.utils.make_grid(predicts)
        self.writer.add_image('ground_truth_%s' % uuid, gt_grid[0], global_step=self.epoch, dataformats='HW')
        self.writer.add_image('predicts_%s' % uuid, predicts_grid[0], global_step=self.epoch, dataformats='HW')

        return np.mean(accu_list), None

def start_train():
    print(uuid)
    # train = TrainModel(args)
    train = TrainModel(args)

    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()