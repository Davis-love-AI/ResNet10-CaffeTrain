#!/bin/sh
export PYTHONPATH=/root/caffe-master/python:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/root/modeltrans_shihl/face_ResNet/trainResNet10
export PYTHONPATH=$PYTHONPATH:/data/public/face/recognition/msceleb/raw

if [ -z "$1" ]
then
    /root/caffe-master/build/tools/caffe train \
        -solver ./train_msceleb_reluadd_hlfinpt.solver
else
    if [ -n "$2" ]
    then
	/root/caffe-master/build/tools/caffe train \
        -solver ./train_msceleb_reluadd_hlfinpt.solver -gpu $1 \
		-snapshot $2
	else
	/root/caffe-master/build/tools/caffe train \
        -solver ./train_msceleb_reluadd_hlfinpt.solver -gpu $1
    fi
fi

