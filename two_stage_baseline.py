from model import Stage1Model, Stage2Model
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from dataset import HelenDataset
from torchvision import transforms
from prepgress import ToTensor, ToPILImage, Resize
from model import ReverseTModel
import torch.optim as optim
import easydict
import torch.nn.functional as F
import argparse
import os
import uuid as uid
from tensorboardX import SummaryWriter

uuid = str(uid.uuid1())[0:8]


class Simple_two(object):
    def __init__(self, batch_size, is_shuffle, num_workers, state_files):
        super(Simple_two, self).__init__()
        self.args = None
        self.get_args()
        self.device = torch.device("cuda:%d" % self.args.cuda if torch.cuda.is_available() else "cpu")
        self.model1 = Stage1Model().to(self.device)
        self.model2 = Stage2Model().to(self.device)
        self.reverse = ReverseTModel().to(self.device)
        self.root_dir = "/data1/yinzi/datas"
        self.predict1 = None
        self.predict2 = None
        self.all_predict = None
        self.best_error = float('Inf')

        self.F1_name_list = ['eyebrow1', 'eyebrow2',
                             'eye1', 'eye2',
                             'nose', 'u_lip', 'i_mouth', 'l_lip']
        self.TP = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.FP = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.TN = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.FN = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}
        self.recall = {x: 0.0 + 1e-20
                       for x in self.F1_name_list}
        self.precision = {x: 0.0 + 1e-20
                          for x in self.F1_name_list}
        self.F1_list = {x: []
                        for x in self.F1_name_list}
        self.F1 = {x: 0.0 + 1e-20
                   for x in self.F1_name_list}

        self.recall_overall_list = {x: []
                                    for x in self.F1_name_list}
        self.precision_overall_list = {x: []
                                       for x in self.F1_name_list}
        self.recall_overall = 0.0
        self.precision_overall = 0.0
        self.F1_overall = 0.0
        self.dataset = None
        self.dataloader = None
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.num_workers = num_workers
        self.state_files = state_files
        self.map_location = self.device
        self.stage2_loss_func = torch.nn.CrossEntropyLoss()
        self.optim2 = optim.Adam(self.model2.parameters(), self.args.lr)
        self.step = 0
        self.epoch = 0
        self.ckpt_dir = "checkpoints_%s" % uuid
        self.writer = SummaryWriter("logs")
        self.get_dataloader()

    def stage1_predict(self, img, label, orig, pretrained=True):
        # if pretrained:
        #     self.load_statefiles(self.state_files['model1'], self.map_location, 'model1')
        self.predict1 = self.model1(img, label, orig)
        return self.predict1

    def stage2_predict(self, parts, pretrained=True):
        # Parts Shape(N, 4, 3, 64, 64)
        # if pretrained:
        #     self.load_statefiles(self.state_files['model2'], self.map_location, 'model2')
        self.predict2 = self.model2(parts)
        return self.predict2

    def get_args(self):
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
        # parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
        # parser.add_argument("--show_freq", default=50, type=int, help="Display image frequency")
        # parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
        # parser.add_argument("--cuda", default=2, type=int, help="Choose GPU with cuda number")
        # parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train")
        # parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
        # self.args = parser.parse_args()
        self.args = easydict.EasyDict({
            "batch_size": 16,
            "display_freq": 10,
            "show_freq": 50,
            "lr": 0.001,
            "cuda": 2,
            "epochs": 50,
            "eval_per_epoch": 1
        })

        print(self.args)
        print(uuid)

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
            state = [torch.load(path[i], map_location=self.map_location)
                     for i in range(4)]

            if isinstance(self.model2, torch.nn.DataParallel):
                self.model2.module.load_state_dict(state[0]['model'])
                best_eye1 = self.model2.module.eye1_model
                self.model2.module.load_state_dict(state[1]['model'])
                best_eye2 = self.model2.module.eye2_model
                self.model2.module.load_state_dict(state[2]['model'])
                best_nose = self.model2.module.nose_model
                self.model2.module.load_state_dict(state[3]['model'])
                best_mouth = self.model2.module.mouth_model
                self.model2.module.eye1_model = best_eye1
                self.model2.module.eye2_model = best_eye2
                self.model2.module.nose_model = best_nose
                self.model2.module.mouth_model = best_mouth

            else:
                self.model2.load_state_dict(state[0]['model'])
                best_eye1 = self.model2.eye1_model
                self.model2.load_state_dict(state[1]['model'])
                best_eye2 = self.model2.eye2_model
                self.model2.load_state_dict(state[2]['model'])
                best_nose = self.model2.nose_model
                self.model2.load_state_dict(state[3]['model'])
                best_mouth = self.model2.mouth_model
                self.model2.eye1_model = best_eye1
                self.model2.eye2_model = best_eye2
                self.model2.nose_model = best_nose
                self.model2.mouth_model = best_mouth

        print('load model from {}'.format(fname))

    def train_stage2(self, pretrained=True):
        if pretrained:
            self.load_statefiles(self.state_files['model1'], self.map_location, 'model1')
            self.load_statefiles(self.state_files['model2'], self.map_location, 'model2')
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            print('Epoch {}/{}'.format(epoch, self.args.epochs - 1))
            print('-' * 10)
            self.model2.train()
            for i_batch, sample_batched in enumerate(self.dataloader['train']):
                self.step += 1
                img = sample_batched['image'].to(self.device)
                labels = sample_batched['labels'].to(self.device)
                orig = {'image': sample_batched['orig'].to(self.device),
                        'label': sample_batched['orig_label'].to(self.device)}
                self.stage1_predict(img, labels, orig, pretrained)
                preds, stage2_parts, stage2_labels = self.predict1
                self.stage2_predict(stage2_parts, pretrained)
                eye1_pred, eye2_pred, nose_pred, mouth_pred = self.predict2
                eye1_loss = self.stage2_loss_func(eye1_pred, stage2_labels[:, 0].long())
                eye2_loss = self.stage2_loss_func(eye2_pred, stage2_labels[:, 1].long())
                nose_loss = self.stage2_loss_func(nose_pred, stage2_labels[:, 2].long())
                mouth_loss = self.stage2_loss_func(mouth_pred, stage2_labels[:, 3].long())
                stage2_loss = eye1_loss + eye2_loss + nose_loss + mouth_loss
                stage2_loss.backward()
                self.optim2.step()
                # statistics
                if self.step % self.args.display_freq == 0:
                    self.writer.add_scalar('loss_stage2_%s' % uuid, stage2_loss.item(), self.step)
                    print('epoch {}\tstep {}\tstage2_loss {:.3}'.format(epoch, self.step, stage2_loss.item()))

            if (epoch + 1) % self.args.eval_per_epoch == 0:
                self.eval()

    def eval(self):
        self.model2.eval()
        with torch.no_grad():
            iter = 0
            stage2_error = 0
            for i, batch in enumerate(self.dataloader['val']):
                iter += 1
                img = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                orig = {'image': batch['orig'].to(self.device),
                        'label': batch['orig_label'].to(self.device)}
                self.stage1_predict(img, labels, orig, pretrained=True)
                preds, stage2_parts, stage2_labels = self.predict1
                self.stage2_predict(stage2_parts, pretrained=True)
                eye1_pred, eye2_pred, nose_pred, mouth_pred = self.predict2
                eye1_loss = self.stage2_loss_func(eye1_pred, stage2_labels[:, 0].long())
                eye2_loss = self.stage2_loss_func(eye2_pred, stage2_labels[:, 1].long())
                nose_loss = self.stage2_loss_func(nose_pred, stage2_labels[:, 2].long())
                mouth_loss = self.stage2_loss_func(mouth_pred, stage2_labels[:, 3].long())
                stage2_error += (eye1_loss + eye2_loss + nose_loss + mouth_loss).item()

            stage2_error /= iter
        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)

        if stage2_error < self.best_error:
            self.best_error = stage2_error
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error_val_%s' % uuid, stage2_error, self.epoch)
        print('epoch {}\terror {:.3}\tbest_error {:.3}'.format(self.epoch, stage2_error, self.best_error))

    def pipline(self, pretrained=True):
        self.load_statefiles(self.state_files['model1'], self.map_location, 'model1')
        self.load_statefiles(self.state_files['model2'], self.map_location, 'model2')
        for i_batch, sample_batched in enumerate(self.dataloader['test']):
            img = sample_batched['image'].to(self.device)
            labels = sample_batched['labels'].to(self.device)
            orig = {'image': sample_batched['orig'].to(self.device),
                    'label': sample_batched['orig_label'].to(self.device)}
            self.stage1_predict(img, labels, orig, pretrained)
            preds, stage2_parts, stage2_labels = self.predict1
            self.stage2_predict(stage2_parts, pretrained)
            # self.all_predict = self.model1.select_net.reverse_transform(preds=self.predict2)
            self.all_predict = self.reverse(preds=self.predict2, theta=self.model1.select_net.theta)
            self.calc_f1(self.all_predict, orig['label'])
            # Shape(N, 1, 512, 512)

            # all_grid = torchvision.utils.make_grid(self.all_predict).detach().cpu()
            # plt.imshow(all_grid[0])
            # print("orig img imshow")
            # orig_grid = torchvision.utils.make_grid(orig['image']).detach().cpu()
            # plt.imshow(orig_grid.permute(1, 2, 0))
            # plt.pause(0.0001)

            # print("Stage1 pred imshow")
            # stage1_pred = torchvision.utils.make_grid(torch.unsqueeze(preds.argmax(dim=1, keepdim=False),
            #                                                           dim=1)).detach().cpu()
            # plt.imshow(stage1_pred[0])
            # plt.pause(0.0001)
            # print("Stage2 predict imshow")
            # for i in range(4):
            #     s2_predict = self.predict2[i].argmax(dim=1, keepdim=True)
            #     s2_predict = torchvision.utils.make_grid(s2_predict.detach().cpu())
            #     plt.imshow(s2_predict[0])
            #     plt.pause(0.0001)

            # print("Stage2 labels imshow")
            # for i in range(4):
            #     plabel_grid = torchvision.utils.make_grid(torch.unsqueeze(stage2_labels[:, i], dim=1).detach().cpu())
            #     plt.imshow(plabel_grid[0])
            #     plt.pause(0.0001)
            # self.predict2
            print("GroundTruth imshow")
            gt_grid = torchvision.utils.make_grid(torch.unsqueeze(orig['label'], dim=1)).detach().cpu()
            plt.imshow(gt_grid[0])
            plt.pause(0.0001)
            print("All predict imshow")
            # np_all = self.all_predict.detach().cpu().numpy()
            # all_pred = (F.softmax(self.all_predict, dim=1) > 0.5).float()
            all_pred = torch.argmax(self.all_predict, dim=1, keepdim=True)
            all_pred = torchvision.utils.make_grid(all_pred.detach().cpu())
            plt.imshow(all_pred[0])
            plt.pause(0.0001)

            print("Selected Parts imshow")
            print(stage2_parts.shape)
            for i in range(4):
                stage2_grid = torchvision.utils.make_grid(stage2_parts[:, i]).detach().cpu()
                plt.imshow(stage2_grid.permute(1, 2, 0))
                plt.pause(0.0001)

        self.output_f1_score()

    def get_dataloader(self):
        img_root_dir = "/data1/yinzi/datas"
        part_root_dir = "/data1/yinzi/facial_parts"
        root_dir = {
            'image': img_root_dir,
            'parts': part_root_dir
        }
        txt_file_names = {
            'train': "exemplars.txt",
            'val': "tuning.txt",
            'test': "testing.txt"

        }

        self.dataset = {x: HelenDataset(txt_file=txt_file_names[x],
                                        root_dir=root_dir['image'],
                                        transform=transforms.Compose([
                                            ToPILImage(),
                                            Resize((64, 64)),
                                            ToTensor()
                                        ])
                                        )
                        for x in ['train', 'val', 'test']
                        }

        self.dataloader = {x: DataLoader(self.dataset[x], batch_size=self.args.batch_size,
                                         shuffle=True, num_workers=16)
                           for x in ['train', 'val', 'test']
                           }

    def calc_f1(self, predict, labels):
        part_name_list = {1: 'eyebrow1', 2: 'eyebrow2', 3: 'eye1', 4: 'eye2',
                          5: 'nose', 6: 'u_lip', 7: 'i_mouth', 8: 'l_lip'}
        pred = predict.argmax(dim=1, keepdim=False).to(self.device)
        # ground = labels.argmax(dim=1, keepdim=False).to(self.device)
        # np_p = pred.detach().cpu().numpy()
        assert pred.shape == (self.all_predict.shape[0], 512, 512)
        ground = labels.long()
        # np_g = ground.detach().cpu().numpy()
        assert ground.shape == pred.shape

        for i in range(1, 9):
            self.TP[part_name_list[i]] += ((pred == i) * (ground == i)).sum().tolist()
            self.TN[part_name_list[i]] += ((pred != i) * (ground != i)).sum().tolist()
            self.FP[part_name_list[i]] += ((pred == i) * (ground != i)).sum().tolist()
            self.FN[part_name_list[i]] += ((pred != i) * (ground == i)).sum().tolist()
        for r in self.F1_name_list:
            self.recall[r] = self.TP[r] / (
                    self.TP[r] + self.FP[r])
            self.precision[r] = self.TP[r] / (
                    self.TP[r] + self.FN[r])
            self.recall_overall_list[r].append(self.recall[r])
            self.precision_overall_list[r].append(self.precision[r])
            self.F1_list[r].append((2 * self.precision[r] * self.recall[r]) /
                                   (self.precision[r] + self.recall[r]))
        return self.F1_list, self.recall_overall_list, self.precision_overall_list

    def output_f1_score(self):
        print("All F1_scores:")
        for x in self.F1_name_list:
            self.recall_overall_list[x] = np.array(self.recall_overall_list[x]).mean()
            self.precision_overall_list[x] = np.array(self.precision_overall_list[x]).mean()
            self.F1[x] = np.array(self.F1_list[x]).mean()
            print("{}:{}\t".format(x, self.F1[x]))
        for x in self.F1_name_list:
            self.recall_overall += self.recall_overall_list[x]
            self.precision_overall += self.precision_overall_list[x]
        self.recall_overall /= len(self.F1_name_list)
        self.precision_overall /= len(self.F1_name_list)
        self.F1_overall = (2 * self.precision_overall * self.recall_overall) / \
                          (self.precision_overall + self.recall_overall)
        print("{}:{}\t".format("overall", self.F1_overall))
        return self.F1, self.F1_overall
# # #
# /home/yinzi/data3
# state_files = {'model1': '/home/yinzi/data3/vimg18/python_projects/new_end2end/checkpoints_1c80dfb0',
#                'model2': '/home/yinzi/data3/vimg18/python_projects/three_face/checkpoint_d02d957f'}
# testrun = Simple_two(batch_size=4, is_shuffle=False, num_workers=4, state_files=state_files)
# # testrun.pipline()
# testrun.train_stage2()
