import copy
import os
import time
from matplotlib import cm
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import NoPriorDataModule
from model import NoPriorVSSLModel
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from sklearn import metrics
import xml.etree.ElementTree as ET
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def convert_to_vggss(filename):
    # prefix = filename[0:11]
    # suffix = int(filename[12:].replace(".mp4", ""))
    # return f"{prefix}_{suffix*100}_{(suffix+10)*1000}"
    return filename.replace(".mp4", "") # .mp4를 제거 ex)video.mp4 -> video

# test의 gt를 처리하는 과정
def testset_gt(args, name):
    if 'flickr' in args.testset:
        gt = ET.parse(args.gt_path + '%s.xml' % name).getroot()
        # check is gt empty
        if len(gt) == 0:
            print('empty gt')
        gt_map = np.zeros([224, 224])
        bboxs = []
        for child in gt:
            for childs in child:
                bbox = []
                if childs.tag == 'bbox':
                    for index, ch in enumerate(childs):
                        if index == 0:
                            continue
                        bbox.append(int(224 * int(ch.text)/256))
                bboxs.append(bbox)
        for item_ in bboxs:
            temp = np.zeros([224, 224])
            (xmin, ymin, xmax, ymax) = item_[0], item_[1], item_[2], item_[3]
            temp[item_[1]:item_[3], item_[0]:item_[2]] = 1
            gt_map += temp
        gt_map /= 2
        gt_map[gt_map > 1] = 1

    elif 'vggss' in args.testset: # testset이 vggss인 경우
        # if gt_all is None:
        if not hasattr(args, 'gt_all'): # args에 gt_all이 없는 경우
            args.gt_all = {} # 초기화
            with open('metadata/vggss.json') as json_file: # vggss.json 파일을 열어 gt load(bbox 정보를 포함하고 있음)
                annotations = json.load(json_file)
            for annotation in annotations: # 모든 데이터 각각에 대해 파일 이름을 key로 하여 bbox를 gt_all에 저장
                args.gt_all[convert_to_vggss(
                    annotation['file'])] = annotation['bbox']
        gt = args.gt_all[name] #'name'에 해당하는 gt 데이터를 gt에 저장
        gt_map = np.zeros([224, 224]) # 초기화
        for item_ in gt: # 각 gt 데이터에 대해 bbox에 해당하는 영역을 1로 설정
            item_ = list(map(lambda x: int(224 * max(x, 0)), item_))
            temp = np.zeros([224, 224])
            (xmin, ymin, xmax, ymax) = item_[0], item_[1], item_[2], item_[3]
            temp[ymin:ymax, xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map > 0] = 1 # gt_map의 값이 0보다 큰 위치는 모두 1로 설정

    elif 'music_duet' in args.testset:
        if not hasattr(args, 'gt_all'):
            args.gt_all = {}
            with open('metadata/music_duet.json') as json_file:
                annotations = json.load(json_file)
            for annotation in annotations:
                bbox = [annotation["bbox_src1"], annotation["bbox_src2"]]
                args.gt_all[annotation['file']] = bbox
        gt = args.gt_all[name]
        gt_map = np.zeros([224, 224])
        for item_ in gt:
            item_ = list(map(lambda x: int(224 * max(x, 0)), item_))
            temp = np.zeros([224, 224])
            (xmin, ymin, xmax, ymax) = item_[0], item_[1], item_[2], item_[3]
            xmax = xmax + xmin
            ymax = ymax + ymin
            temp[ymin:ymax, xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map > 0] = 1

    elif 'music_solo' in args.testset:
        if not hasattr(args, 'gt_all'):
            args.gt_all = {}
            with open('metadata/music_solo.json') as json_file:
                annotations = json.load(json_file)
            for annotation in annotations:
                bbox = [annotation["bbox"]]
                args.gt_all[annotation['file']] = bbox
        gt = args.gt_all[name]
        gt_map = np.zeros([224, 224])
        for item_ in gt:
            item_ = list(map(lambda x: int(224 * max(x, 0)), item_))
            temp = np.zeros([224, 224])
            (xmin, ymin, xmax, ymax) = item_[0], item_[1], item_[2], item_[3]
            xmax = xmax + xmin
            ymax = ymax + ymin
            temp[ymin:ymax, xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map > 0] = 1
    return gt_map


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='unspecified',
                        type=str, help='experiment name')
    parser.add_argument('--trainset', default='vggss',
                        type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss',
                        type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='',
                        type=str, help='Root directory path of data')
    parser.add_argument('--test_data_path', default='',
                        type=str, help='Root directory path of data')
    parser.add_argument('--image_size', default=224,
                        type=int, help='Height and width of inputs')
    parser.add_argument('--gt_path', default='', type=str)
    parser.add_argument('--epsilon', default=0.65, type=float)
    parser.add_argument('--epsilon_margin', default=0.25, type=float)
    parser.add_argument('--logit_temperature', default=0.07, type=float)
    parser.add_argument('--trimap', default=1, type=int)
    parser.add_argument('--pretrain_flow', default=1, type=int)
    parser.add_argument('--pretrain_vision', default=1, type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--flow', type=int, default=1)
    parser.add_argument('--flowtype', type=str, default='cnn')
    parser.add_argument('--freeze_vision', type=int, default=1)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--ckpt', default='', type=str,
                        help='filepath of weights')
    parser.add_argument('--description', default='',
                        type=str, help='description of experiment')
    parser.add_argument('--concat_num', default=2, type=int)
    parser.add_argument('--visualize', default=False, type=bool)

    return parser.parse_args()

# 이미지 정규화
def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin # value의 최소값을 vmin으로 설정
    vmax = value.max() if vmax is None else vmax # value의 최대값을 vmax로 설정
    if not (vmax - vmin) == 0: # vmin과 vmax가 같지 않다면
        value = (value - vmin) / (vmax - vmin)  # 정규화 --> [0,1] 범위로 스케일링

    return value


class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224)) # 초기화
        infer_map[infer >= thres] = 1 # inference 값이 threshold보다 크거나 같을 경우 1로 설정
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) +
                                          np.sum(infer_map * (gtmap == 0))) # ciou 계산
        self.ciou.append(ciou) # self.ciou에 추가
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap == 0))) # ciou, 분자, 분모 반환

    def final(self):
        ciou = np.mean(np.array(self.ciou)) # self.ciou에 저장된 모든 ciou 값의 평균을 반환/최종 성능 평가
        return ciou

    def clear(self):
        self.ciou = [] # 초기화


class NoPriorModule(LightningModule):
    def __init__(self, args):
        super(NoPriorModule, self).__init__()
        self.args = args

        self.model = NoPriorVSSLModel(self.args)

        self.evaluator = Evaluator()

        self.iou = []

        self.ap = []

        self.cap = []

        self.true = []

        self.object_number = [] # 추가

        if args.freeze_vision: # 만약 visual encoder에 freezing 되어 있다면
            print("FREEZING IMAGE ENCODER")
            self.model.unfreeze_vision(False)

        # gt for vggss
        if args.testset == 'vggss': # test set이 vggss라면
            args.gt_all = {} # gt_all 초기화
            with open('metadata/vggss.json') as json_file: # vggss.json 파일을 열어 gt load(bbox 정보를 포함하고 있음)
                annotations = json.load(json_file)
            for annotation in annotations: # 모든 데이터 각각에 대해 파일 이름을 key로 하여 bbox를 gt_all에 저장
                args.gt_all[convert_to_vggss(
                    annotation['file'])] = annotation['bbox']

        # gt for music_duet
        if args.testset == 'music_duet':
            args.gt_all = {}
            with open('metadata/music_duet.json') as json_file:
                annotations = json.load(json_file)
            for annotation in annotations:
                bbox = [annotation["bbox_src1"], annotation["bbox_src2"]]
                args.gt_all[annotation['file']] = bbox

        self.save_hyperparameters()

    # 모델의 학습 과정 정의, 학습 batch에 대한 연산 수행
    def training_step(self, batch, batch_idx):
        image, flow, spec, file_id, _ = batch # 배치 데이터 추출

        loss, attention = self.model(
            image.float(), flow.float(), spec.float(), True)

        self.log("train/loss", loss.item(), prog_bar=True,
                 on_epoch=True, batch_size=self.args.batch_size)

        return loss

    # 모델의 검증 과정 정의
    def validation_step(self, batch, batch_idx):
        image, flow, spec, file_id, _ = batch # 배치 데이터 추출

        file_id = file_id[0].split("+")

        loss, localization_arr = self.model(
            image.float(), flow.float(), spec.float(), False)

        if localization_arr == []:
            return {"loss": loss, "localization": localization_arr}

        localization_ = torch.stack(
            localization_arr, dim=0).sum(dim=0).clone().detach()

        localization_ = localization_.squeeze(0).cpu().numpy()
        localization_map = cv2.resize(localization_, dsize=(
            224, 224 * self.args.concat_num), interpolation=cv2.INTER_CUBIC)

        localization_map = np.split(
            localization_map, [224 * i for i in range(1, self.args.concat_num)], axis=0)

        gt_arr = []

        for i in range(self.args.concat_num):
            sample_gt = testset_gt(self.args, file_id[i])

            gt_arr.append(sample_gt)

        for localization in localization_arr:

            localization = localization.squeeze(0).cpu().numpy()
            localization_map = cv2.resize(localization, dsize=(
                224, 224 * self.args.concat_num), interpolation=cv2.INTER_CUBIC)

            localization_map = np.split(
                localization_map, [224 * i for i in range(1, self.args.concat_num)], axis=0)

            test_iou = []
            test_cap = []

            for i in range(self.args.concat_num):
                sample_localization = normalize_img(localization_map[i])

                sample_gt = gt_arr[i]

                thr = np.sort(sample_localization.flatten())[
                    int(sample_localization.shape[0] * sample_localization.shape[1] / 2)]

                sample_localization[sample_localization > thr] = 1
                sample_localization[sample_localization < 1] = 0
                self.evaluator = Evaluator()
                ciou_lvs, inter, union = self.evaluator.cal_CIOU(
                    sample_localization, sample_gt, 0.5)

                test_iou.append(ciou_lvs)

            iou_argmax = np.argmax(test_iou) # test_iou의 최대값 인덱스 찾기

            self.iou.append(test_iou[iou_argmax]) # iou의 최대값 추가
            # self.cap.append(test_cap[iou_argmax])
            gt_arr[iou_argmax] = np.zeros_like(gt_arr[iou_argmax])

        if len(localization_arr) < self.args.concat_num:
            for i in range(self.args.concat_num - len(localization_arr)):
                self.iou.append(0)
                # self.cap.append(0)

        # localization_ = torch.stack(localization_arr, dim=0).mean(dim=0)

        if len(localization_arr) == self.args.concat_num:
            self.true.append(1)
        else:
            self.true.append(0)

        return {"loss": loss, "localization": localization}

    def test_step(self, batch, batch_idx):
        image, flow, spec, file_id, _ = batch

        save_file_id = file_id[0]

        file_id = file_id[0].split("+")

        loss, localization_arr = self.model(
            image.float(), flow.float(), spec.float(), False)

        if len(localization_arr) == self.args.concat_num:
            self.true.append(1)
        else:
            self.true.append(0)

        if localization_arr == []:
            return {"loss": loss, "localization": localization_arr}

        localization_ = torch.stack(
            localization_arr, dim=0).sum(dim=0)

        localization_ = localization_.squeeze(0).cpu().numpy()
        localization_map = cv2.resize(localization_, dsize=(
            224, 224 * self.args.concat_num), interpolation=cv2.INTER_CUBIC)

        localization_map = np.split(
            localization_map, [224 * i for i in range(1, self.args.concat_num)], axis=0)

        binary_classification_results = []
        ground_truth_binary = []

        for i in range(self.args.concat_num):
            sample_localization = normalize_img(localization_map[i])

            sample_gt = testset_gt(self.args, file_id[i])

            thr = np.sort(sample_localization.flatten())[
                int(sample_localization.shape[0] * sample_localization.shape[1] / 2)]

            sample_localization[sample_localization > thr] = 1
            sample_localization[sample_localization < 1] = 0

            # Flatten the arrays for pixel-wise comparison.
            predicted_pixels = sample_localization.flatten()
            ground_truth_pixels = sample_gt.flatten()

            # 각 픽셀이 객체인지 배경인지에 대한 예측 값을 binary_classification_results에 추가합니다.
            binary_classification_results.extend(predicted_pixels)

            # 각 픽셀의 ground truth 값을 ground_truth_binary에 추가합니다.
            ground_truth_binary.extend(ground_truth_pixels)

        # sklearn의 average_precision_score 함수를 이용해 pixel-wise AP를 계산합니다.
        pixel_wise_ap = average_precision_score(
            ground_truth_binary, binary_classification_results)

        self.ap.append(pixel_wise_ap)

        gt_arr = []

        for i in range(self.args.concat_num):
            sample_gt = testset_gt(self.args, file_id[i])

            gt_arr.append(sample_gt)

        for localization in localization_arr:

            localization = localization.squeeze(0).cpu().numpy()
            localization_map = cv2.resize(localization, dsize=(
                224, 224 * self.args.concat_num), interpolation=cv2.INTER_CUBIC)

            localization_map = np.split(
                localization_map, [224 * i for i in range(1, self.args.concat_num)], axis=0)

            test_iou = []
            test_cap = []

            for i in range(self.args.concat_num):
                sample_localization = normalize_img(localization_map[i])

                sample_gt = gt_arr[i]

                thr = np.sort(sample_localization.flatten())[
                    int(sample_localization.shape[0] * sample_localization.shape[1] / 2)]

                sample_localization[sample_localization > thr] = 1
                sample_localization[sample_localization < 1] = 0
                self.evaluator = Evaluator()
                ciou_lvs, inter, union = self.evaluator.cal_CIOU(
                    sample_localization, sample_gt, 0.5)

                predicted_pixels = sample_localization.flatten()
                ground_truth_pixels = sample_gt.flatten()

                pixel_wise_ap = average_precision_score(
                    ground_truth_pixels, predicted_pixels)

                test_iou.append(ciou_lvs)

                test_cap.append(pixel_wise_ap)

            iou_argmax = np.argmax(test_iou)

            self.iou.append(test_iou[iou_argmax])
            self.cap.append(test_cap[iou_argmax])
            gt_arr[iou_argmax] = np.zeros_like(gt_arr[iou_argmax])

        if len(localization_arr) < self.args.concat_num:
            for i in range(self.args.concat_num - len(localization_arr)):
                self.iou.append(0)
                self.cap.append(0)

        return {"loss": loss, "localization": localization}

    def on_validation_epoch_end(self):

        results = []
        for i in range(21):
            result = np.sum(np.array(self.iou) >= 0.05 * i)
            result = result / len(self.iou)
            results.append(result)
        x = [0.05 * i for i in range(21)]
        auc_ = metrics.auc(x, results)
        thresholds = np.arange(0.5, 1.0, 0.05)
        ciou_values = []
        ciou_0_5 = np.sum(np.array(self.iou) >= 0.5)/len(self.iou)
        ciou_0_3 = np.sum(np.array(self.iou) >= 0.3)/len(self.iou)
        for threshold in thresholds:
            ciou = np.sum(np.array(self.iou) >= threshold) / len(self.iou)
            ciou_values.append(ciou)
        mciou = np.mean(ciou_values)
        self.iou = []
        mtrue = np.mean(self.true)

        self.log("val/cIoU0.3", ciou_0_3, prog_bar=True,
                 on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val/cIoU", ciou_0_5, prog_bar=True,
                 on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val/AUC", auc_, prog_bar=True,
                 on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val/true", mtrue, prog_bar=True,
                 on_epoch=True, batch_size=1, sync_dist=True)

        self.model.update_epoch()

        print("epoch: ", self.model.epoch)

        print("Epoch Time: ", time.time() - self.model.start_time)

        self.model.start_time = time.time()

        self.evaluator = Evaluator()

    def on_test_epoch_end(self):

        results = []
        for i in range(21):
            result = np.sum(np.array(self.iou) >= 0.05 * i)
            result = result / len(self.iou)
            results.append(result)
        x = [0.05 * i for i in range(21)]
        auc_ = metrics.auc(x, results)
        # thresholds = np.arange(0.5, 1.0, 0.05)
        # ciou_values = []
        ciou_0_5 = np.sum(np.array(self.iou) >= 0.5)/len(self.iou)
        ciou_0_3 = np.sum(np.array(self.iou) >= 0.3)/len(self.iou)
        # for threshold in thresholds:
        #     ciou = np.sum(np.array(self.iou) >= threshold) / len(self.iou)
        #     ciou_values.append(ciou)
        # mciou = np.mean(ciou_values)

        piap_ = np.mean(self.ap)
        cap_ = np.mean(self.cap)

        self.iou = []

        self.log("test/cIoU", ciou_0_5, prog_bar=True,
                 on_epoch=True, batch_size=1)
        self.log("test/cIoU0.3", ciou_0_3, prog_bar=True,
                 on_epoch=True, batch_size=1, sync_dist=True)
        self.log("test/CAP", cap_, prog_bar=True, on_epoch=True, batch_size=1)
        self.log("test/PIAP", piap_, prog_bar=True,
                 on_epoch=True, batch_size=1)
        self.log("test/AUC", auc_, prog_bar=True, on_epoch=True, batch_size=1)

        print("True: ", np.mean(self.true))

        self.evaluator = Evaluator()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer


if __name__ == "__main__":
    args = get_arguments()

    torch.set_float32_matmul_precision('high')

    module = NoPriorModule(args)

    if args.ckpt != "":
        if '.pth' in args.ckpt:
            print("Auto Load from model_base")
        else:
            # module.model.aud_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), padding=(1, 1), stride=(1, 1))
            module.load_from_checkpoint(args.ckpt)
        print(module)

        print("Module Loaded!")

    datamodule = NoPriorDataModule(args)

    checkpointer = ModelCheckpoint(dirpath=f"logs/{args.name}/",
                                   filename='best',
                                   monitor="val/cIoU",
                                   save_last=True,
                                   save_weights_only=False,
                                   mode='max')
    # checkpointer = ModelCheckpoint(dirpath=f"logs/{args.name}/",
    #                                 filename='epoch{epoch}-val_cIoU{val_cIoU:.4f}',
    #                                 monitor="val/cIoU",
    #                                 save_top_k=-1,  # Save all checkpoints
    #                                 save_last=True,
    #                                 save_weights_only=False,
    #                                 mode='max')

    logger = TensorBoardLogger("logs", name=args.name)

    # trainer = Trainer(accelerator='gpu',
    #                     devices='auto',
    #                     strategy="ddp_find_unused_parameters_false",
    #                     max_epochs=args.epochs,
    #                     num_sanity_val_steps=400,
    #                     callbacks=[checkpointer],
    #                     logger=logger)

    trainer = Trainer(accelerator='gpu',
                      devices="auto",
                      #   strategy="ddp_find_unused_parameters_true",
                      strategy="ddp_find_unused_parameters_true",
                      max_epochs=args.epochs,
                      num_sanity_val_steps=400,
                      callbacks=[checkpointer],
                      logger=logger)

    trainer.fit(module, datamodule)
