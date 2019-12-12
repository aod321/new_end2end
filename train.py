from model import Stage1Model, Stage2Model
from dataset import HelenDataset
from prepgress import Resize, ToPILImage, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import os
import uuid

uuid = str(uuid.uuid1())[0:8]


class TrainTest(object):
    def __init__(self):
        super(TrainTest, self).__init__()
        self.args = None
        self.dataset = None
        self.data_loader = None
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')
        self.get_args()
        self.device = torch.device("cuda:%d" % self.args.cuda if torch.cuda.is_available() else "cpu")
        self.get_dataloader()
        self.model1 = Stage1Model().to(self.device)
        self.model2 = Stage2Model().to(self.device)
        self.optim1 = optim.Adam(self.model1.parameters(), self.args.lr)
        # self.optim2 = optim.Adam(self.model2.parameters(), self.args.lr)
        self.optim2 = optim.Adam(self.model2.parameters(), self.args.lr)
        self.stage1_loss_func = nn.CrossEntropyLoss()
        self.stage2_loss_func = nn.CrossEntropyLoss()
        self.writer = SummaryWriter('runs')
        self.ckpt_dir = "checkpoints_%s" % uuid

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
        parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
        parser.add_argument("--show_freq", default=50, type=int, help="Display image frequency")
        parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
        parser.add_argument("--cuda", default=2, type=int, help="Choose GPU with cuda number")
        parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train")
        parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
        self.args = parser.parse_args()
        print(self.args)
        print(uuid)

    def get_dataloader(self):
        img_root_dir = "/data1/yinzi/datas"
        part_root_dir = "/data1/yinzi/facial_parts"
        root_dir = {
            'image': img_root_dir,
            'parts': part_root_dir
        }
        txt_file_names = {
            'train': "exemplars.txt",
            'val': "tuning.txt"
        }

        self.dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                                        root_dir=root_dir['image'],
                                        transform=transforms.Compose([
                                            ToPILImage(),
                                            Resize((512, 512)),
                                            ToTensor()
                                        ])
                                        )
                        for x in ['train', 'val']
                        }

        self.data_loader = {x: DataLoader(self.dataset[x], batch_size=self.args.batch_size,
                                          shuffle=True, num_workers=16)
                            for x in ['train', 'val']
                            }

    def start_train(self):

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            print('Epoch {}/{}'.format(epoch, self.args.epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            self.model1.train()  # Set model to training mode
            self.model2.train()

            for i, batch in enumerate(self.data_loader['train']):
                self.step += 1
                image, label = batch['image'].to(self.device), batch['labels'].to(self.device)
                orig = {'image': batch['orig'].to(self.device),
                        'label': batch['orig_label'].to(self.device)}
                # Image Shape(N, 3, 64, 64)
                # zero the parameter gradients
                self.optim1.zero_grad()
                self.optim2.zero_grad()
                # Train Stage1
                # Label Shape(N, 512, 512)
                preds, parts, parts_label = self.model1(image, label, orig)
                # Preds Shape(N, 9, 512, 512)
                # Parts Shape(N, 4, 3, 64, 64)
                # Parts_labels Shape(N, 4, 64, 64) Cross
                stage1_mask_loss = self.stage1_loss_func(preds, label.long())
                stage1_mask_loss.backward()
                self.optim1.step()
                # Train Stage2
                # Stage2_preds Shape(N, 8, 64, 64)
                stage2_preds = self.model2(parts)
                eye1_pred, eye2_pred, nose_pred, mouth_pred = stage2_preds
                eye1_loss = self.stage2_loss_func(eye1_pred, parts_label[:, 0].long())
                eye2_loss = self.stage2_loss_func(eye2_pred, parts_label[:, 1].long())
                nose_loss = self.stage2_loss_func(nose_pred, parts_label[:, 2].long())
                mouth_loss = self.stage2_loss_func(mouth_pred, parts_label[:, 3].long())
                stage2_loss = eye1_loss + eye2_loss + nose_loss + mouth_loss
                stage2_loss.backward()
                self.optim2.step()
                loss = stage1_mask_loss.item() + stage2_loss.item()
                # statistics
                if self.step % self.args.display_freq == 0:
                    self.writer.add_scalar('loss_stage1_%s' % uuid, stage1_mask_loss.item(), self.step)
                    self.writer.add_scalar('loss_stage2_%s' % uuid, stage2_loss.item(), self.step)
                    print('epoch {}\tstep {}\tstage1_loss {:.3}\tstage2_loss {:.3}'
                          '\n all_loss{:.3}'.format(
                        epoch, self.step, stage1_mask_loss.item(), stage2_loss.item(), loss))

            # pred_arg = preds.argmax(dim=1, keepdim=False)
            # binary_list = []
            # for i in range(preds.shape[1]):
            #     binary = (pred_arg == i).float()
            #     binary_list.append(binary)
            # pred_onehot = torch.stack(binary_list, dim=1)
            # if self.step % self.args.show_freq == 0:
            #     print("show images on tensorboard")
            #     for o in range(1, 9):
            #         preds_grid = torchvision.utils.make_grid(torch.unsqueeze(pred_onehot[:, o], dim=1))
            #         self.writer.add_image('train_Stage1Preds_%s_%d' % (uuid, o), preds_grid, self.step)
            #     for o in range(4):
            #         parts_grid = torchvision.utils.make_grid(parts[:, o])
            #         self.writer.add_image('train_Stage1Select_%s_%d' % (uuid, o), parts_grid, self.step)
            #         plabels_grid = torchvision.utils.make_grid(torch.unsqueeze(parts_label[:, o], dim=1))
            #         self.writer.add_image('train_Stage1plabels_%s_%d' % (uuid, o), plabels_grid, self.step)
            #     print("All images showed")

            if (epoch + 1) % self.args.eval_per_epoch == 0:
                self.eval()

    def eval(self):
        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            error = 0
            iter = 0
            stage1_error = 0
            stage2_error = 0
            for i, batch in enumerate(self.data_loader['val']):
                iter += 1
                image = batch['image'].to(self.device)
                label = batch['labels'].to(self.device)
                orig = {'image': batch['orig'].to(self.device),
                        'label': batch['orig_label'].to(self.device)}
                preds, parts, parts_label = self.model1(image, label, orig)
                stage1_mask_loss = self.stage1_loss_func(preds, label.long())
                stage1_error += stage1_mask_loss.item()
                stage2_preds = self.model2(parts)
                eye1_pred, eye2_pred, nose_pred, mouth_pred = stage2_preds
                eye1_loss = self.stage2_loss_func(eye1_pred, parts_label[:, 0].long())
                eye2_loss = self.stage2_loss_func(eye2_pred, parts_label[:, 1].long())
                nose_loss = self.stage2_loss_func(nose_pred, parts_label[:, 2].long())
                mouth_loss = self.stage2_loss_func(mouth_pred, parts_label[:, 3].long())
                stage2_loss = eye1_loss.item() + eye2_loss.item() + nose_loss.item() + mouth_loss.item()
                stage2_error += stage2_loss
                error += stage1_mask_loss.item() + stage2_loss

            error /= iter
            stage1_error /= iter
            stage2_error /= iter
        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)
        if error < self.best_error:
            self.best_error = error
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error_val_%s' % uuid, error, self.epoch)
        self.writer.add_scalar('error_stage1_%s' % uuid, stage1_mask_loss.item(), self.epoch)
        self.writer.add_scalar('error_stage2_%s' % uuid, stage2_loss, self.epoch)
        print('epoch {}\terror {:.3}\tbest_error {:.3}\n'
              'stage1_error {:.3}\tstage2_error{:.3}'.format(self.epoch, error, self.best_error,
                                                             stage1_error, stage2_error))

        # ++++++ Show image +++++
        # Preds Shape(N, 9, 512, 512)
        # Parts Shape(N, 8, 3, 64, 64)
        # Parts_labels Shape(N, 4, 64, 64) Binary
        # Stage2_preds Shape(N, 8, 64, 64)
        pred_arg = preds.argmax(dim=1, keepdim=False)
        binary_list = []
        for i in range(preds.shape[1]):
            binary = (pred_arg == i).float()
            binary_list.append(binary)
        pred_onehot = torch.stack(binary_list, dim=1)
        print("show images on tensorboard")
        for o in range(1, 9):
            preds_grid = torchvision.utils.make_grid(torch.unsqueeze(pred_onehot[:, o], dim=1))
            self.writer.add_image('val_Stage1preds_%s_%d' % (uuid, o), preds_grid, self.epoch)
        for o in range(4):
            parts_grid = torchvision.utils.make_grid(parts[:, o])
            self.writer.add_image('val_SelectedParts_%s_%d' % (uuid, o), parts_grid, self.epoch)
            plabels_grid = torchvision.utils.make_grid(torch.unsqueeze(parts_label[:, o], dim=1))
            self.writer.add_image('val_plabels_%s_%d' % (uuid, o), plabels_grid, self.epoch)
            # eye1_pred, eye2_pred, nose_pred, mouth_pred
            eye1_pred = (torch.softmax(eye1_pred, dim=1) > 0.5).float()
            eye2_pred = (torch.softmax(eye2_pred, dim=1) > 0.5).float()
            nose_pred = (torch.softmax(nose_pred, dim=1) > 0.5).float()
            mouth_pred = (torch.softmax(mouth_pred, dim=1) > 0.5).float()
            eye1_grid = torchvision.utils.make_grid(torch.unsqueeze((1 - eye1_pred[:, 0]), dim=1))
            eye2_grid = torchvision.utils.make_grid(torch.unsqueeze((1 - eye2_pred[:, 0]), dim=1))
            nose_grid = torchvision.utils.make_grid(torch.unsqueeze((1 - nose_pred[:, 0]), dim=1))
            mouth_grid = torchvision.utils.make_grid(torch.unsqueeze((1 - mouth_pred[:, 0]), dim=1))
            self.writer.add_image('eye1_pred%s_%d' % (uuid, o), eye1_grid, self.epoch)
            self.writer.add_image('eye2_pred%s_%d' % (uuid, o), eye2_grid, self.epoch)
            self.writer.add_image('nose_pred%s_%d' % (uuid, o), nose_grid, self.epoch)
            self.writer.add_image('mouth_pred%s_%d' % (uuid, o), mouth_grid, self.epoch)

        print("All images showed")

        torch.cuda.empty_cache()

    def save_state(self, fname, optim=True):
        state = {}

        if isinstance(self.model1, torch.nn.DataParallel):
            state['model1'] = self.model1.module.state_dict()
        else:
            state['model1'] = self.model1.state_dict()

        if isinstance(self.model2, torch.nn.DataParallel):
            state['model2'] = self.model2.module.state_dict()
        else:
            state['model2'] = self.model2.state_dict()

        if optim:
            state['optim1'] = self.optim1.state_dict()
            state['optim2'] = self.optim2.state_dict()

        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))


testrain = TrainTest()
testrain.start_train()
