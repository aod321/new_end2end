import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch.optim as optim
from template import TemplateModel, F1Accuracy
from model import Stage2Model, Stage1Model, ReverseTModel
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
parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate for optimizer")
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

        self.model = Stage2Model().to(self.device)
        self.model1 = Stage1Model().to(self.device)
        # self.reverse = ReverseTModel().to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.metric = nn.CrossEntropyLoss()
        self.metric = F1Accuracy()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_loader = stage1_dataloaders['train']
        self.eval_loader = stage1_dataloaders['val']

        self.ckpt_dir = "checkpoints_%s" % uuid
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        image, label = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)
        orig = {'image': batch['orig'].to(self.device),
                'label': batch['orig_label'].to(self.device)}
        preds, stage2_parts, stage2_labels = self.model1(image, label, orig)
        stage2_preds = self.model(stage2_parts)
        loss = self.criterion(all_predict, orig['label'].long())
        # loss = self.criterion(pred, y)
        # loss = self.criterion(preds, label.long())
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
            image, label = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)
            orig = {'image': batch['orig'].to(self.device),
                    'label': batch['orig_label'].to(self.device)}
            preds, stage2_parts, stage2_labels = self.model1(image, label, orig)
            stage_preds = self.model(stage2_parts)
            all_predict = self.reverse(preds=stage_preds, theta=self.model1.select_net.theta)
            error = self.metric(all_predict, orig['label'].long())
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

    def load_statefiles(self, fname, map_location, model='model1'):
        if model == 'model1':
            path = os.path.join(fname, 'best.pth.tar')
            state = torch.load(path, map_location=map_location)
            if isinstance(self.model1, torch.nn.DataParallel):
                self.model1.module.load_state_dict(state['model'])
            else:
                self.model1.load_state_dict(state['model'])

        if model == 'model2':
            path = [os.path.join(fname, 'best_%s.pth.tar' % x)
                    for x in ['eye1', 'eye2', 'nose', 'mouth']]
            state = [torch.load(path[i], map_location=map_location)
                     for i in range(4)]

            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(state[0]['model'])
                best_eye1 = self.model.module.eye1_model
                self.model.module.load_state_dict(state[1]['model'])
                best_eye2 = self.model.module.eye2_model
                self.model.module.load_state_dict(state[2]['model'])
                best_nose = self.model.module.nose_model
                self.model.module.load_state_dict(state[3]['model'])
                best_mouth = self.model.module.mouth_model
                self.model.module.eye1_model = best_eye1
                self.model.module.eye2_model = best_eye2
                self.model.module.nose_model = best_nose
                self.model.module.mouth_model = best_mouth

            else:
                self.model.load_state_dict(state[0]['model'])
                best_eye1 = self.model.eye1_model
                self.model.load_state_dict(state[1]['model'])
                best_eye2 = self.model.eye2_model
                self.model.load_state_dict(state[2]['model'])
                best_nose = self.model.nose_model
                self.model.load_state_dict(state[3]['model'])
                best_mouth = self.model.mouth_model
                self.model.eye1_model = best_eye1
                self.model.eye2_model = best_eye2
                self.model.nose_model = best_nose
                self.model.mouth_model = best_mouth

        print('load model from {}'.format(fname))

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
            image, label = batch['image'].float().to(self.device), batch['labels'].float().to(self.device)
            orig = {'image': batch['orig'].to(self.device),
                    'label': batch['orig_label'].to(self.device)}
            preds, stage2_parts, stage2_labels = self.model1(image, label, orig)
            stage_preds = self.model(stage2_parts)
            all_predict = self.reverse(preds=stage_preds, theta=self.model1.select_net.theta)
            all_predict = torch.squeeze(all_predict, dim=1)
            all_predict = F.one_hot(all_predict.long(), num_classes=-1).to(self.device)
            all_predict = all_predict.permute(0, 3, 1, 2)
            accu = self.metric(all_predict, orig['label'].long())
            accu_list.append(accu)

        predicts = all_predict.argmax(dim=1, keepdim=False)
        predicts = torch.unsqueeze(predicts, dim=1)
        predicts_grid = torchvision.utils.make_grid(predicts)
        self.writer.add_image('predicts_%s' % uuid, predicts_grid[0], global_step=self.epoch, dataformats='HW')

        return np.mean(accu_list), None


def start_train():
    state_files = {'model1': '/home/yinzi/data3/vimg18/python_projects/new_end2end/checkpoints_1c80dfb0',
                   'model2': '/home/yinzi/data3/vimg18/python_projects/three_face/checkpoint_d02d957f'}
    print(uuid)
    # train = TrainModel(args)
    train = Train_F1_eval(args)
    train.load_statefiles(state_files['model1'], train.device, 'model1')
    train.load_statefiles(state_files['model2'], train.device, 'model2')
    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
