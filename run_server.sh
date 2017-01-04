#!/bin/bash

python ./faster_rcnn/server.py --model ./lib/pretrained/Resnet_iter_200000.ckpt --net Resnet50_test --gpu 0

