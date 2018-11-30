#!/bin/sh
export PYTHONPATH=/root/caffe-master/python:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/root/modeltrans_shihl/face_ResNet/trainResNet10
export PYTHONPATH=$PYTHONPATH:/data/public/face/recognition/msceleb/raw

if [ -z "$1" ]
then
    /root/caffe-master/build/tools/caffe train \
        -solver ./train_msceleb_reluadd.solver
else
    /root/caffe-master/build/tools/caffe train \
        -solver ./train_msceleb_reluadd.solver -gpu $1
    if [ -n "$2" ]
    then
       \  -snapshot ./model_snapshot/solver_msceleb_iter_$2.solverstate
    fi
fi

