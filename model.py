import numpy as np
import model_base
import torch
import torch.nn.functional as F
from torch import nn
import copy
from PIL import Image
import time


class NoPriorVSSLModel(model_base.NoPriorVSSLModel):
    def __init__(self, args):
        super().__init__(args)
        self.epoch = -1
        self.start_time = time.time()
        # self.fix_flow_att()

        # print model size
        def count_parameters(model: nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model size: ", count_parameters(self))

    # 현재 epoch 수를 업데이트
    def update_epoch(self):
        self.epoch += 1

    def attention(self, img, aud_vect, flow, att, train=None):
        attention, _ = att(img, flow) # img, flow attention

        # if train:
        #     attendedimg = nn.functional.normalize(img + attention, dim=1)
        # else:
        #     attendedimg = nn.functional.normalize(img, dim=1)

        attendedimg = nn.functional.normalize(img + attention, dim=1)

        return self.lvs_loss(attendedimg, aud_vect, train)

    def forward(self, image, flow, audio, train=None):
        return self.adaptive_forward(image, flow, audio, train)

    def adaptive_forward(self, image, flow, audio, train=None):
        B, C, H, W = image.shape

        feat_H, feat_W = (H // 32), (W // 32)

        norm_img = self.img_normalize(image)
        img = self.imgnet(norm_img).view(-1, 512, feat_H, feat_W) # visual feature [-1, 512, 7, 7]
        aud = self.audnet(audio).view(-1, 512, 9, 9) # audio feature

        aud_vect = self.aud_max_pool(aud).squeeze(-1).squeeze(-1)
        aud_vect = nn.functional.normalize(aud_vect, dim=1)

        img_vect = self.aud_max_pool(img).squeeze(-1).squeeze(-1)
        img_vect = nn.functional.normalize(img_vect, dim=1)

        if self.flowtype == 'cnn':
            flow = self.flownet(flow).view(-1, 512, feat_H, feat_W) # flow feature
        elif self.flowtype == 'maxpool':
            flow = self.flownet(flow)

        loss1, localization1_ = self.attention(
            img, aud_vect, flow, self.flowatt, train)
        # loss1, localization1_ = self.attention(
        #     img, img_vect, flow, self.flowatt, train)

        return loss1, localization1_
