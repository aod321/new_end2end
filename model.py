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
        one_preds = (F.softmax(preds, dim=1) > 0.5).float()
        cens = calc_centroid(one_preds) * 8  # Shape(N, 9, 2)
        # eye1 = cens[:, 3]
        # eye2 = cens[:, 4]
        eye1 = (cens[:, 1] + cens[:, 3]) / 2.0
        eye2 = (cens[:, 2] + cens[:, 4]) / 2.0
        nose = cens[:, 5]
        mouth = (cens[:, 6] + cens[:, 7] + cens[:, 8]) / 3.0
        points = torch.stack([eye1, eye2, nose, mouth], dim=1)  # Shape(N, 4, 2)
        if orig:
            parts, parts_label = self.select_net(orig['image'], orig['label'], points)
        else:
            parts, parts_label = self.select_net(img, labels, points)

        return preds, parts, parts_label

    # Preds Shape(N, 9, 512, 512)
    # Parts Shape(N, 4, 3, 64, 64)
    # Parts_labels Shape(N, 4, 64, 64) CrossEn


class SelectNet(nn.Module):
    def __init__(self):
        super(SelectNet, self).__init__()
        self.theta = None
        self.rtheta = None
        self.points = None
        self.device = None

    def get_theta(self):
        self.device = self.points.device
        # points in [N, 4, 2]
        points_in = self.points
        # print(points_in)
        N = points_in.shape[0]
        param = torch.zeros((N, 4, 2, 3)).to(self.device)
        for i in range(2):
            points_in[:, i, 0] = 256 - 8 * points_in[:, i, 0]
            points_in[:, i, 1] = 256 - 8 * points_in[:, i, 1]
            param[:, i, 0, 0] = 8
            param[:, i, 0, 2] = points_in[:, i, 1]
            param[:, i, 1, 1] = 8
            param[:, i, 1, 2] = points_in[:, i, 0]

        for i in range(2, 4):
            points_in[:, i, 0] = 256 - 6.4 * points_in[:, i, 0]
            points_in[:, i, 1] = 256 - 6.4 * points_in[:, i, 1]
            param[:, i, 0, 0] = 6.4
            param[:, i, 0, 2] = points_in[:, i, 1]
            param[:, i, 1, 1] = 6.4
            param[:, i, 1, 2] = points_in[:, i, 0]
        # Param Shape(N, 4, 2, 3)
        # Every label has a affine param
        ones = torch.tensor([[0., 0., 1.]]).repeat(N, 4, 1, 1).to(self.device)
        param = torch.cat([param, ones], dim=2)
        param = torch.inverse(param)
        # ---               ---
        # Then, convert all the params to thetas
        self.theta = torch.zeros([N, 4, 2, 3]).to(self.device)
        self.theta[:, :, 0, 0] = param[:, :, 0, 0]
        self.theta[:, :, 0, 1] = param[:, :, 0, 1]
        self.theta[:, :, 0, 2] = param[:, :, 0, 2] * 2 / 512 + self.theta[:, :, 0, 0] + self.theta[:, :, 0, 1] - 1
        self.theta[:, :, 1, 0] = param[:, :, 1, 0]
        self.theta[:, :, 1, 1] = param[:, :, 1, 1]
        self.theta[:, :, 1, 2] = param[:, :, 1, 2] * 2 / 512 + self.theta[:, :, 1, 0] + self.theta[:, :, 1, 1] - 1
        # theta Shape(N, 4, 2, 3)
        return self.theta

    def reverse_transform(self, preds):
        N = self.theta.shape[0]
        ones = torch.tensor([[0., 0., 1.]]).repeat(N, 4, 1, 1).to(self.device)
        self.rtheta = torch.cat([self.theta, ones], dim=2).to(self.device)
        self.rtheta = torch.inverse(self.rtheta)
        self.rtheta = self.rtheta[:, :, 0:2]
        assert self.rtheta.shape == (N, 4, 2, 3)
        del ones
        eye1_pred, eye2_pred, nose_pred, mouth_pred = preds
        # Parts_pred argmax Shape(N, 64, 64)
        eye1_pred = eye1_pred.argmax(dim=1, keepdim=False).float()
        eye2_pred = eye2_pred.argmax(dim=1, keepdim=False).float()
        nose_pred = nose_pred.argmax(dim=1, keepdim=False).float()
        mouth_pred = mouth_pred.argmax(dim=1, keepdim=False).float()
        eye1_pred[eye1_pred == 1] = 1
        eye1_pred[eye1_pred == 2] = 3

        eye2_pred[eye2_pred == 2] = 4
        eye2_pred[eye2_pred == 1] = 2
        # np_eye2_pred = eye2_pred.detach().cpu().numpy()

        nose_pred[nose_pred == 1] = 5

        mouth_pred[mouth_pred == 1] = 6
        mouth_pred[mouth_pred == 2] = 7
        mouth_pred[mouth_pred == 3] = 8
        predicts = torch.stack([eye1_pred, eye2_pred, nose_pred, mouth_pred], dim=1)
        # del eye1_pred, eye2_pred, nose_pred, mouth_pred
        assert predicts.shape == (N, 4, 64, 64)

        sample = torch.zeros((N, 1, 512, 512)).to(self.device)
        for i in range(4):
            grid = F.affine_grid(theta=self.rtheta[:, i], size=[N, 1, 512, 512], align_corners=True).to(self.device)
            pred = torch.unsqueeze(predicts[:, i], dim=1).to(self.device)
            sample += F.grid_sample(input=pred, grid=grid, mode='nearest', align_corners=True)
            # sample[sample>8] = 0
            # del grid, pred
        # np_sample = sample.detach().cpu().numpy()
        # del predicts
        return sample

    def forward(self, img, label, points):
        self.points = points.clone()
        theta = self.get_theta()
        self.device = img.device
        n, l, h, w = img.shape
        samples = []
        labels = []
        for i in range(4):
            grid = F.affine_grid(theta[:, i], [n, l, 64, 64], align_corners=True).to(img.device)
            samples.append(F.grid_sample(input=img, grid=grid, align_corners=True))
            labels.append(F.grid_sample(input=torch.unsqueeze(label, dim=1), mode='nearest',grid=grid, align_corners=True))
        samples = torch.stack(samples, dim=0)
        samples = samples.transpose(1, 0)
        labels = torch.cat(labels, dim=1)
        assert samples.shape == (n, 4, 3, 64, 64)
        assert labels.shape == (n, 4, 64, 64)
        # 单个成分labels 的Shape都是(N, 64, 64) ，以下程序处理序号以方便CrossEntropy训练
        eye1_labels = labels[:, 0]
        eye2_labels = labels[:, 1]
        nose_labels = labels[:, 2]
        mouth_labels = labels[:, 3]

        eye1_labels[(eye1_labels != 3) * (eye1_labels != 1)] = 0  # bg
        eye1_labels[eye1_labels == 1] = 1  # eyebrow1
        eye1_labels[eye1_labels == 3] = 2  # eye1

        eye2_labels[(eye2_labels != 4) * (eye2_labels != 2)] = 0  # bg
        eye2_labels[eye2_labels == 2] = 1  # eyebrow2
        eye2_labels[eye2_labels == 4] = 2  # eye2

        nose_labels[nose_labels != 5] = 0  # bg
        nose_labels[nose_labels == 5] = 1  # nose

        mouth_labels[(mouth_labels != 6) * (mouth_labels != 7) * (mouth_labels != 8)] = 0  # bg
        mouth_labels[mouth_labels == 6] = 1  # up_lip
        mouth_labels[mouth_labels == 7] = 2  # in_mouth
        mouth_labels[mouth_labels == 8] = 3  # down_lip

        labels = torch.stack([eye1_labels, eye2_labels, nose_labels, mouth_labels], dim=1)
        assert labels.shape == (img.shape[0], 4, 64, 64)

        return samples, labels


class Stage2Model(nn.Module):
    def __init__(self):
        super(Stage2Model, self).__init__()
        self.eye1_model = EyeModel()
        self.eye2_model = EyeModel()
        self.nose_model = NoseModel()
        self.mouth_model = MouthModel()

    def forward(self, parts):
        eye1 = parts[:, 0]
        eye2 = parts[:, 1]
        nose = parts[:, 2]
        mouth = parts[:, 3]
        # Shape(N, 3, 64, 64)
        out_eye1 = self.eye1_model(eye1)
        out_eye2 = self.eye1_model(eye2)
        out_nose = self.nose_model(nose)
        out_mouth = self.mouth_model(mouth)

        return out_eye1, out_eye2, out_nose, out_mouth


class PartsModel(nn.Module):
    def __init__(self):
        super(PartsModel, self).__init__()
        self.model = Stage2FaceModel()

    def forward(self, x):
        # x Shape (N, 3, 64, 64)
        out = self.model(x)
        return out


class EyeModel(PartsModel):
    def __init__(self):
        super(EyeModel, self).__init__()
        self.model.set_label_channels(3)


class NoseModel(PartsModel):
    def __init__(self):
        super(NoseModel, self).__init__()
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
        ones = torch.tensor([[0., 0., 1.]]).repeat(N, 4, 1, 1).to(self.device)
        self.rtheta = torch.cat([theta, ones], dim=2).to(self.device)
        self.rtheta = torch.inverse(self.rtheta)
        self.rtheta = self.rtheta[:, :, 0:2]
        assert self.rtheta.shape == (N, 4, 2, 3)
        del ones
        # Parts_pred argmax Shape(N, 64, 64)
        fg = []
        bg = []
        for i in range(4):
            all_pred = F.softmax(preds[i], dim=1)
            grid = F.affine_grid(theta=self.rtheta[:, i], size=[N, preds[i].shape[1], 512, 512], align_corners=True).to(
                self.device)
            bg_grid = F.affine_grid(theta=self.rtheta[:, i], size=[N, 1, 512, 512], align_corners=True).to(
                self.device)
            temp = F.grid_sample(input=all_pred, grid=grid, mode='nearest', padding_mode='zeros', align_corners=True)
            temp2 = F.grid_sample(input=all_pred[:, 0:1], grid=bg_grid, mode='nearest', padding_mode='border', align_corners=True)
            bg.append(temp2)
            fg.append(temp[:, 1:])
            # del temp, temp2
        bg = torch.cat(bg, dim=1)
        bg = (bg[:, 0] * bg[:, 1] * bg[:, 2] * bg[:, 3]).unsqueeze(dim=1)
        fg = torch.cat(fg, dim=1)  # Shape(N, 8, 512 ,512)
        sample = torch.cat([bg, fg[:, 0:1], fg[:, 2:3], fg[:, 1:2], fg[:, 3:]], dim=1)  # Shape(N, 9, 512, 512)


        # np_sample = sample.detach().cpu().numpy()
        assert sample.shape == (N, 9, 512, 512)
        return sample


class TwoStagePipeLine(nn.Module):
    def __init__(self):
        super(TwoStagePipeLine, self).__init__()
        self.stage1_model = Stage1Model()
        self.stage2_model = Stage2Model()

    def forward(self, img, label):
        preds, parts, parts_label = self.stage1_model(img, label)
        eye1, eye2, nose, mouth = parts[:, 0], parts[:, 1], parts[:, 2], parts[:, 3]
        eye1_pred, eye2_pred, nose_pred, mouth_pred = self.stage2_model(eye1, eye2, nose, mouth)

        return preds, eye1_pred, eye2_pred, nose_pred, parts_label
