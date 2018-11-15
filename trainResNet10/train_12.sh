#!/usr/bin/env sh
export PYTHONPATH=/home/shengcn/caffe/python
export PYTHONPATH=$PYTHONPATH:/home/shengcn/projects/facedetection/train/12net

if [ -z "$1" ]
then
    /home/shengcn/caffe/build/tools/caffe train \
        -solver ./solver_12.prototxt
else
    /home/shengcn/caffe/build/tools/caffe train \
        -solver ./solver_12.prototxt \
        -snapshot ./model_snapshot/solver_12_iter_$1.solverstate
fi

#
#set -e
#~/caffe/build/tools/caffe train \
#	 --solver=./solver.prototxt \
#  	 --weights=./24net-only-cls.caffemodel
