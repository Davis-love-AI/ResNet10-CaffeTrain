#!/bin/sh
export PYTHONPATH=/root/caffe-master/python:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/shihl/.bashrcface_ResNet/trainResNet10
export PYTHONPATH=$PYTHONPATH:/data/public/face/recognition/msceleb/raw
export PYTHONPATH=$PYTHONPATH:/data-face/xp_data

if [ -z "$1" ]
then
    /root/caffe-master/build/tools/caffe train \
        -solver ./train_msceleb_reluadd.solver
elif [ -n "$2" ]
then
    /root/caffe-master/build/tools/caffe train \
        -solver ./train_msceleb_reluadd.solver -gpu $1 \
        -snapshot $2
else
    /root/caffe-master/build/tools/caffe train \
        -solver ./train_msceleb_reluadd.solver -gpu $1
fi

