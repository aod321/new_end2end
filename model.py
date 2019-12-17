import torch.nn as nn
from icnnmodel import Stage2FaceModel, FaceModel
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def calc_centroid(tensor):
    # Inputs Shape(N, 9 , 64, 64)
    # Return Shape(N, 9 ,2)
    input = tensor.float() + 1e-10
    n, l, h, w = input.shape
    indexs_y = torch.from_numpy(np.arange(h)).float().to(tensor.device)
    indexs_x = torch.from_numpy(np.arange(w)).float().to(tensor.device)
    center_y = input.sum(3) * indexs_y.view(1, 1, -1)
    center_y = center_y.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    center_x = input.sum(2) * indexs_x.view(1, 1, -1)
    center_x = center_x.sum(2, keepdim=True) / input.sum([2, 3]).view(n, l, 1)
    # output = torch.cat([center_y, center_x], 2)
    output = torch.cat([center_y, center_x], 2)
    return output


class Stage1Model(nn.Module):
    def __init__(self):
        super(Stage1Model, self).__init__()
        self.model = FaceModel()
        self.select_net = SelectNet()

    def forward(self, img, labels, orig=None):
        # Image Shape(N, 3, 512, 512)
        preds = self.model(img)
        # Preds Shape(N, 9, 512, 512)
        # Calc centroids
        big_preds = (F.interpolate(F.softmax(preds, dim=1), (512, 512)) > 0.5).float()
        # one_preds = (F.softmax(preds, dim=1) > 0.5).float()
        cens = calc_centroid(big_preds)  # Shape(N, 9, 2)
        mouth = (cens[:, 6:7] + cens[:, 7:8] + cens[:, 8:9]) / 3.0
        points = torch.cat([cens[:, 1:6], mouth], dim=1)  # Shape(N, 6, 2)
        assert points.shape == (img.shape[0], 6, 2)
        if orig:
            parts, parts_label = self.select_net(orig['image'], orig['label'], points)
        else:
            parts, parts_label = self.select_net(img, labels, points)

        return preds, parts, parts_label

    # Preds Shape(N, 9, 512, 512)
    # Parts Shape(N, 6, 3, 64, 64)
    # Parts_labels Shape(N, 6, 64, 64) CrossEn


class SelectNet(nn.Module):
    def __init__(self):
        super(SelectNet, self).__init__()
        self.theta = None
        self.points = None
        self.device = None

    def forward(self, img, label, points):
        # self.points = points[:, 1:]  # Shape(N, 8, 2)
        self.points = points
        self.device = img.device
        n, l, h, w = img.shape
        # self.points = torch.cat([self.points[:, 0:6],
        #                          self.points[:, 6:9].mean(dim=1, keepdim=True)],
        #                         dim=1)
        # assert self.points.shape == (n, 6, 2)
        theta = torch.zeros((n, 6, 2, 3)).to(self.device)
        # Crop brow1 brow2 eye1 eye2 nose
        for i in range(5):
            theta[:, i, 0, 0] = 64 / w
            theta[:, i, 0, 2] = -1 + (2 * self.points[:, i, 1]) / w
            theta[:, i, 1, 1] = 64 / h
            theta[:, i, 1, 2] = -1 + (2 * self.points[:, i, 0]) / h
        # Crop mouth
        for i in range(5, 6):
            theta[:, i, 0, 0] = 80 / w
            theta[:, i, 0, 2] = -1 + (2 * self.points[:, i, 1]) / w
            theta[:, i, 1, 1] = 80 / h
            theta[:, i, 1, 2] = -1 + (2 * self.points[:, i, 0]) / h
        self.theta = theta
        samples = []
        labels = []
        for i in range(6):
            grid = F.affine_grid(theta[:, i], [n, l, 64, 64], align_corners=True).to(img.device)
            samples.append(F.grid_sample(input=img, grid=grid, align_corners=True))
            labels.append(
                F.grid_sample(input=torch.unsqueeze(label, dim=1), mode='nearest', grid=grid, align_corners=True))
        samples = torch.stack(samples, dim=0).transpose(1, 0)
        labels = torch.cat(labels, dim=1)
        assert samples.shape == (n, 6, 3, 64, 64)
        assert labels.shape == (n, 6, 64, 64)
        # 单个成分labels 的Shape都是(N, 64, 64) ，以下程序处理序号以方便CrossEntropy训练
        mouth_labels = labels[:, 5:6]
        mouth_labels[(mouth_labels != 6) * (mouth_labels != 7) * (mouth_labels != 8)] = 0  # bg
        mouth_labels[mouth_labels == 6] = 1  # up_lip
        mouth_labels[mouth_labels == 7] = 2  # in_mouth
        mouth_labels[mouth_labels == 8] = 3  # down_lip
        for i in range(5):
            labels[:, i][labels[:, i] != i+1] = 0
            labels[:, i][labels[:, i] == i+1] = 1

        labels = torch.cat([labels[:, 0:5], mouth_labels], dim=1)
        assert labels.shape == (img.shape[0], 6, 64, 64)

        return samples, labels


class Stage2Model(nn.Module):
    def __init__(self):
        super(Stage2Model, self).__init__()
        self.single_model = nn.ModuleList([SingleModel()
                                           for _ in range(5)])
        self.mouth_model = MouthModel()

    def forward(self, parts):
        # Shape(N, 3, 64, 64)
        out = []
        for i in range(5):
            out.append(self.single_model[i](parts[:, i]))
        out_mouth = self.mouth_model(parts[:, 5])   # (N, 4, 64, 64)
        out.append(out_mouth)
        return out


class PartsModel(nn.Module):
    def __init__(self):
        super(PartsModel, self).__init__()
        self.model = Stage2FaceModel()

    def forward(self, x):
        # x Shape (N, 3, 64, 64)
        out = self.model(x)
        return out


class SingleModel(PartsModel):
    def __init__(self):
        super(SingleModel, self).__init__()
        self.model.set_label_channels(2)


class MouthModel(PartsModel):
    def __init__(self):
        super(MouthModel, self).__init__()
        self.model.set_label_channels(4)


class ReverseTModel(nn.Module):
    def __init__(self):
        super(ReverseTModel, self).__init__()
        self.device = None
        self.rtheta = None

    def forward(self, preds, theta):
        self.device = theta.device
        N = theta.shape[0]
        ones = torch.tensor([[0., 0., 1.]]).repeat(N, 6, 1, 1).to(self.device)
        self.rtheta = torch.cat([theta, ones], dim=2).to(self.device)
        self.rtheta = torch.inverse(self.rtheta)
        self.rtheta = self.rtheta[:, :, 0:2]
        assert self.rtheta.shape == (N, 6, 2, 3)
        del ones
        # Parts_pred argmax Shape(N, 64, 64)
        fg = []
        bg = []
        for i in range(6):
            all_pred = F.softmax(preds[i], dim=1)
            grid = F.affine_grid(theta=self.rtheta[:, i], size=[N, preds[i].shape[1], 512, 512], align_corners=True).to(
                self.device)
            bg_grid = F.affine_grid(theta=self.rtheta[:, i], size=[N, 1, 512, 512], align_corners=True).to(
                self.device)
            temp = F.grid_sample(input=all_pred, grid=grid, mode='nearest', padding_mode='zeros', align_corners=True)
            temp2 = F.grid_sample(input=all_pred[:, 0:1], grid=bg_grid, mode='nearest', padding_mode='border',
                                  align_corners=True)
            bg.append(temp2)
            fg.append(temp[:, 1:])
            # del temp, temp2
        bg = torch.cat(bg, dim=1)
        bg = (bg[:, 0:1] * bg[:, 1:2] * bg[:, 2:3] * bg[:, 3:4] *
              bg[:, 4:5] * bg[:, 5:6])
        fg = torch.cat(fg, dim=1)  # Shape(N, 8, 512 ,512)
        sample = torch.cat([bg, fg], dim=1)
        assert sample.shape == (N, 9, 512, 512)
        return sample


class TwoStagePipeLine(nn.Module):
    def __init__(self):
        super(TwoStagePipeLine, self).__init__()
        self.stage1_model = Stage1Model()
        self.stage2_model = Stage2Model()

    def forward(self, img, label):
        preds, parts, parts_label = self.stage1_model(img, label)
        stage2_preds = self.stage2_model(parts)
        return preds, stage2_preds
