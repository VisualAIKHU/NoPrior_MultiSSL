import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
import math
import copy

# x가 속한 집합의 대표 찾기
def find(parent, x):
    if parent[x] != x: # x의 parent가 x가 아니라면
        parent[x] = find(parent, parent[x]) # find를 통해 재귀적으로 x의 parent를 찾음
    return parent[x] # x의 대표 반환

# x와 y가 속한 집합을 합치기
def union(parent, x, y):
    rootX = find(parent, x) # x의 대표 찾기
    rootY = find(parent, y) # y의 대표 찾기
    if rootX != rootY: # x와 y의 대표가 다른 경우(다른 집합인 경우)
        parent[rootY] = rootX # y의 대표를 x의 대표로 교체(y의 집합을 x의 집합에 통합)


class NoPriorVSSLModel(nn.Module):
    def __init__(self, args):
        super(NoPriorVSSLModel, self).__init__()
        self.args = args
        self.tau = self.args.tau
        self.flowtype = self.args.flowtype
        self.freeze_vision = self.args.freeze_vision
        self.trimap = self.args.trimap
        self.pretrain_flow = True if self.args.pretrain_flow else False
        self.pretrain_vision = True if self.args.pretrain_vision else False
        self.logit_temperature = self.args.logit_temperature
        self.object_loss_step = -1

        self.img_normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])

        # Vision model
        self.imgnet = resnet18(pretrained=self.pretrain_vision)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()

        self.dist = nn.PairwiseDistance(p=2)

        # Audio model
        self.audnet = resnet18()
        # Fix first layer channel
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.Identity()
        self.audnet.fc = nn.Identity()

        self.aud_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Flow model
        if self.flowtype == 'cnn':
            self.flownet = resnet18(pretrained=self.pretrain_flow)
            # Fix first layer channel
            self.flownet.conv1 = nn.Conv2d(2, 64, kernel_size=(
                7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.flownet.avgpool = nn.Identity()
            self.flownet.fc = nn.Identity()
            self.flowatt = Self_Attn(512, 512)
        elif self.flowtype == 'maxpool':
            self.flownet = nn.AdaptiveMaxPool2d((7, 7))
            self.flowatt = Self_Attn(512, 2)

        self.m = nn.Sigmoid()
        self.epsilon = self.args.epsilon # 0.65
        self.epsilon2 = self.args.epsilon - self.args.epsilon_margin # 0.4

        for m in self.audnet.modules(): # audnet의 모든 레이어에 대해
            if isinstance(m, nn.Conv2d): # Conv2d가 있다면
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu') # 해당 방식으로 초기화
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): # BatchNorm2d, GroupNorm이 있다면
                nn.init.normal_(m.weight, mean=1, std=0.02) # 해당 방식으로 초기화(weight)
                nn.init.constant_(m.bias, 0) # 해당 방식으로 초기화(bias)

        if args.ckpt is not None and args.ckpt != "": # args.ckpt가 비어있지 않다면
            if '.pth' in args.ckpt: # args.ckpt에 .pth가 포함된다면
                weights = torch.load(args.ckpt) # args.ckpt에서 weights 로드/즉 사전 학습된 모델을 불러올 때 사용
                self.load_state_dict(weights['model'])

        # print model size
        def count_parameters(model: nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model size: ", count_parameters(self))

    # imgnet의 parameter의 가중치 고정 또는 해제
    def unfreeze_vision(self, grad):
        for param in self.imgnet.parameters(): # imgnet의 모든 param에 대해
            param.requires_grad = grad

    # 두 객체 간의 cosine similarity 계산
    def cosine_sim_object(self, obj1, obj2):
        return F.cosine_similarity(obj1.squeeze(2), obj2.squeeze(2), dim=1)

    # lvs loss 계산
    def lvs_loss(self, img, aud, train=None):
        B, C, H, W = img.shape
        self.mask = (1 - 100 * torch.eye(B, B)).to(img.device) # 대각요소가 -99, 나머지는 1인 행렬 생성

        PosLogits = torch.einsum(
            'ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)  # torch.Size([128, 1, 7, 7]) / sound-associated map Sv
        Alllogits = torch.einsum(
            'ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])  # torch.Size([128, 512, 7, 7]) / Poslogit에서 차원평균을 하지 않은 것

        if self.trimap:
            Pos2 = self.m((PosLogits - self.epsilon2)/self.tau) # negative를 제외한 나머지
            Neg = 1 - Pos2 # 실제 negative 부분
        else:
            Neg = 1 - Pos

        # Normal LVS Loss
        # return 0, torch.Tensor(PosLogits.squeeze(1)).view(1, 1, H, W)

        thr = (PosLogits * Neg).max() + 0.1

        logit = img.view(B, C, -1) * PosLogits.view(B, 1, -1) # visual feature Fv * Sv = Sound-Associated Visual Feature Fv_hat
        positive = PosLogits.clone().view(B, -1) # PosLogit(Sv)를 복사한 후 2차원 텐서로 변환
        neg_val = (logit * Neg.view(B, 1, -1)).mean(dim=2) # Neg는 채널정보가 없어서 채널정보가 있는 negative를 대표하는 값을 만들기 위해 logit이랑 곱함

        #초기화
        endpoint = torch.zeros(B, dtype=torch.bool).to(img.device)
        already_pos = torch.zeros_like(positive, dtype=torch.bool)

        object_loss = 0
        loop = 0

        # Make Object_Arr to save object, shape is [0, 512]
        object_arr = torch.Tensor([]).to(img.device) # 비어있는 1차원 텐서 생성

        object_dict = [{} for _ in range(B)] # 배치 크기 B의 리스트 생성, 리스트 내의 각 요소는 dictionary

        localization_map = [] # 비어있는 리스트 생성

        while (True):
            # Get the argmax locations / Sv에서 최대값의 index를 찾음
            pos_max_indices = positive.argmax(dim=1).detach()

            # Get the values of the logits at the argmax locations / 선택된 인덱스에 해당하는 Fv_hat 추출(Ep)
            selected_pos_values = logit[torch.arange(
                B), :, pos_max_indices].view(B, 512, 1)

            # torch.Size([N, B, 512]) / object array에 Ep 저장
            object_arr = torch.cat(
                (object_arr, (selected_pos_values.squeeze(2) * ~endpoint.unsqueeze(1)).unsqueeze(0)), dim=0)

            # torch.Size([B, W*H]) / positive region(Rp)
            dist_to_logit = self.dist(logit.permute(
                0, 2, 1), selected_pos_values.permute(0, 2, 1))

            # localization_map = localization_map + -1 * dist_to_logit
            localization_map.append(-1 * dist_to_logit.detach())

            # torch.Size([B, W*H]) / negative region(Rn)
            dist_to_zero = self.dist(logit.permute(
                0, 2, 1), neg_val.unsqueeze(1))

            pos = dist_to_logit < dist_to_zero
            pos = pos * ~already_pos
            already_pos = already_pos + pos
            positive = positive * (1 - pos.float())

            # if self.epoch >= self.object_loss_step:
            #     object_loss += (self.object_loss(dist_to_logit,
            #                                      pos.view(B, -1)) * endpoint.float()).mean()

            # print("object_loss: ", object_loss)

            for i in range(B):
                if pos[i].sum() == 0 or positive[i].max() < 0.1: # 배치에 더 이상 처리할 pos가 없거나 0.1 이하로 떨어지면
                    endpoint[i] = True # 해당 배치 완료

            if endpoint.sum() == B: # 모든 배치가 완료되면
                break # 루프 종료(조건문, 반복문 모두)

            loop += 1 # 루프 반복 횟수 증가


        N = object_arr.shape[0] # object bank에 저장된 cell의 개수

        for i in range(B):
            parent = list(range(N)) # N개의 초기 parent 설정

            neg = neg_val[i]  # torch.Size([512]) / 현재 배치에 대한 neg
            batch = object_arr[:, i]  # torch.Size([N, 512]) / 현재 배치에 대한 object bank에 저장된 cell

            # neg와 각 객체 간의 유사도 계산 -> N개
            neg_sim = F.cosine_similarity(neg.unsqueeze(0), batch)

            # print("neg_sim: ", neg_sim)

            thr1 = 0.7 # 1
            thr2 = 0.6 # 1

            # neg와 유사도가 높은 객체를 처리
            for j in range(N):
                if neg_sim[j] > thr1: # cosine similarity가 thr1보다 크다면
                    if -1 not in object_dict[i]: # object_dict에 -1이라는 key가 존재하지 않는다면
                        object_dict[i][-1] = [j] # 새로운 키를 생성하고 value에 객체의 인덱스 j 할당
                    else: # 존재한다면
                        object_dict[i][-1].append(j) # 인덱스 j 추가. 즉, neg와 유사한 객체들을 계속해서 추가하는 과정임
                    continue  # 이미 neg와 유사도가 높은 객체는 다른 작업을 수행하지 않고 계속한다.(해당 조건문 종료, 다음 반복문 수행)

                # 나머지 객체들에 대한 유사도 처리
                for k in range(j):
                    sim = F.cosine_similarity(
                        batch[j].unsqueeze(0), batch[k].unsqueeze(0)) # 나머지 객체들과의 cosine similarity 계산
                    if sim > thr2: # 만약 thr2보다 크다면
                        union(parent, j, k) # 두 객체를 하나의 그룹으로 통합

            # 부모가 같은 객체들을 같은 그룹으로 분류
            for j in range(N):
                if neg_sim[j] > thr1:
                    continue  # 이미 neg와 유사도가 높은 객체는 건너뛴다.

                root = find(parent, j) # 해당 객체의 parent인 root(대표)를 찾음
                if root in object_dict[i]: # object_dict에 root key가 존재한다면
                    object_dict[i][root].append(j) # 해당 객체 추가
                else: # 존재하지 않는다면
                    object_dict[i][root] = [j] # 새로운 그룹을 시작하고 해당 객체 추가

        # print("dict len: ", len(set(object_dict[0].keys()) - set([-1])))

        # assert False

        Pos = self.m((PosLogits - self.epsilon)/self.tau) # 실제 positive 부분

        if train:

            loss = self.lvs(B, img, Pos, PosLogits, Alllogits)

            object_loss = self.object_loss(object_arr, object_dict, neg_val)

            total = loss + object_loss

            return total, PosLogits.squeeze(1)

        # get localization map which is no inside object_dict[0][-1]
        localization_map = torch.cat(localization_map, dim=1)
        localization_map = localization_map.view(N, 1, H, W)

        # return 0, torch.Tensor(PosLogits.squeeze(1)).view(1, 1, H, W)

        # return 0, localization_map

        localization_out = [] # 초기화
        for key in object_dict[0].keys(): # object_dict의 모든 key에 대해
            if key == -1: # -1이라면
                continue # 조건문 나와서 다음번 반복문 진행
            # print("key: ", key)
            # print("object_dict[0][key]: ", object_dict[0][key])
            localization_out.append(localization_map[object_dict[0]
                                                     [key]].mean(dim=0).view(1, H, W)) # 평균내어 맵 추출

        return 0, localization_out

    def lvs(self, B, img, Pos, PosLogits, Alllogits):
        if self.trimap:
            Pos2 = self.m((PosLogits - self.epsilon2)/self.tau) # 실제 negative 이외의 영역(positive+uncertain)
            Neg = 1 - Pos2 # 실제 negative
        else:
            Neg = 1 - Pos

        # print("Neg Value", Neg.max(), Neg.min(), Neg.mean(), Neg.std())

        PosAll = self.m((Alllogits - self.epsilon)/self.tau)

        sim1 = (Pos * PosLogits).view(*
                                      PosLogits.shape[:2], -1).sum(-1) / (Pos.view(*Pos.shape[:2], -1).sum(-1))
        sim = ((PosAll * Alllogits).view(*Alllogits.shape[:2], -1).sum(-1) / PosAll.view(
            *PosAll.shape[:2], -1).sum(-1)) * self.mask
        sim2 = (Neg * PosLogits).view(*
                                      PosLogits.shape[:2], -1).sum(-1) / (Neg.view(*Neg.shape[:2], -1).sum(-1))

        logits = torch.cat((sim1, sim, sim2), 1)/self.logit_temperature

        target = torch.zeros((B), dtype=torch.long).to(img.device)

        loss = F.cross_entropy(logits, target)

        return loss

    def object_loss(self, obj_arr, obj_dict, neg_val):
        loss = 0.0 # 초기화
        # obj_arr: torch.Size([N, B, 512])
        # obj_dict: [{}, {}, ...]
        obj_arr = obj_arr.transpose(0, 1)  # B N 512
        for i in range(len(obj_dict)): # 그룹개수만큼 중 i번째 그룹에 대해
            for key in obj_dict[i]: # i번째 그룹의 key에 대해
                if key == -1: # key가 -1이라면(background라면)
                    continue # 해당 조건문 종료, 다음 반복문 실행(단지 넘어가는 용도)

                # Anchor 벡터
                anchor = obj_arr[i, key]
                # continue if anchor is with zero vector
                if anchor.sum() == 0:
                    continue

                # Positive 벡터 - key에 해당하는 벡터들의 평균을 구한다
                positive_indices = obj_dict[i][key]
                positive = obj_arr[i, positive_indices].mean(dim=0)

                # Negative 벡터 - key에 해당하지 않는 벡터들의 합을 구한다
                negative_indices = list(
                    set(range(obj_arr.shape[1])) - set(positive_indices))
                negative = obj_arr[i, negative_indices].mean(
                    dim=0) + neg_val[i]

                # if negative.sum == 0 or nan
                if negative.sum() == 0 or torch.isnan(negative).sum() > 0:
                    continue

                # 정규화
                anchor = F.normalize(anchor, dim=0)
                positive = F.normalize(positive, dim=0)
                negative = F.normalize(negative, dim=0)

                # 유사도 구하기
                pos_total = torch.matmul(anchor, positive.t())
                neg_total = torch.matmul(anchor, negative.t())
                # print("pos_total: ", pos_total, "neg_total: ", neg_total)

                loss += (1.0 - pos_total) + neg_total

                # print("loss : ", loss)

                # assert False

        loss = loss / len(obj_dict)

        return loss  # / 10

    # def object_loss(self, logits, pos):
    #     pos_logit = logits * pos  # B W*H
    #     neg_logit = logits * (~pos)  # B W*H

    #     pos_logit = pos_logit.sum(dim=1)
    #     neg_logit = neg_logit.sum(dim=1)

    #     numerator = torch.exp(-1 * pos_logit / 20)
    #     denominator = torch.exp(-1 * neg_logit / 20) + \
    #         torch.exp(-1 * pos_logit / 20)

    #     # print(f"numerator: {numerator}, denominator: {denominator}")

    #     loss = -torch.log(numerator/denominator + 1e-8)

    #     return loss * 100

    def forward(self, image, flow, audio):

        # Audio
        audn = self.audnet(audio).view(-1, 512, 9, 9) # audio feature Fa
        aud = self.aud_max_pool(audn).squeeze(-1).squeeze(-1) # GAP Ia
        aud = nn.functional.normalize(aud, dim=1) # Ia normalize

        # Image
        norm_img = self.img_normalize(image) # image normalize
        img = self.imgnet(norm_img).view(-1, 512, 7, 7) # visual feature Fv

        # Flow
        if self.flowtype == 'cnn':
            flow = self.flownet(flow).view(-1, 512, 7, 7) # flow feature Ff
        elif self.flowtype == 'maxpool':
            flow = self.flownet(flow)

        # Cross visual-flow attention
        attention, _ = self.flowatt(img, flow)

        attendedimg = img + attention
        attendedimg = nn.functional.normalize(attendedimg, dim=1)

        loss1, localization1_ = self.lvs_loss(attendedimg, aud)

        return loss1, localization1_


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, key_in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=key_in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, v):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(v).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key) / \
            math.sqrt(self.chanel_in//8)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        return out, attention
