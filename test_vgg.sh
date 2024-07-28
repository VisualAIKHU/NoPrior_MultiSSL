#!/bin/bash

echo "NoPrior Testing Script"

name="last_train_vgg3_v0"
ckpt=""
trainset="vggss"
testset="vggss"
train_data_path=""
test_data_path=""
gt_path=""
image_size=224
batch_size=128
concat_num=3

CUDA_VISIBLE_DEVICES=6 python3 test.py --name "$name" --ckpt "$ckpt" --trainset "$trainset" --testset "$testset" --train_data_path "$train_data_path" --test_data_path "$test_data_path" --image_size "$image_size" --batch_size "$batch_size" --gt_path "$gt_path" --concat_num "$concat_num"


