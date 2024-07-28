#!/bin/bash

echo "NoPrior Training Script"

name="music"
description=""
trainset="music_solo"
testset="music_solo"
train_data_path=""
test_data_path=""
gt_path=""
image_size=224
epochs=100
batch_size=128
freeze_vision=1
lr=0.0001
ckpt=""
concat_num=1

# With CKPT
# CUDA_VISIBLE_DEVICES=6 python3 train.py --name "$name" --trainset "$trainset" --testset "$testset" --train_data_path "$train_data_path" --test_data_path "$test_data_path" --image_size "$image_size" --epochs "$epochs" --batch_size "$batch_size" --gt_path "$gt_path" --freeze_vision "$freeze_vision" --lr "$lr" --ckpt "$ckpt" --description "$description"

# Without CKPT
CUDA_VISIBLE_DEVICES=4,5 python3 train.py --name "$name" --trainset "$trainset" --testset "$testset" --train_data_path "$train_data_path" --test_data_path "$test_data_path" --image_size "$image_size" --epochs "$epochs" --batch_size "$batch_size" --gt_path "$gt_path" --freeze_vision "$freeze_vision" --lr "$lr" --description "$description" --concat_num "$concat_num"
