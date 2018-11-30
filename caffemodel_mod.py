import caffe
import numpy as np
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_def",
        default="/home/zhangxin/github/caffe/models/bvlc_googlenet/deploy.prototxt",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default="/home/zhangxin/github/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel",
        help="Trained model weights file."
    )
#    parser.add_argument(
#        "--gpu",
#        action='store_true',
#        help="Switch for gpu computation."
#    )
    return parser.parse_args()

def main(args):
#    if args.gpu:
#        caffe.set_mode_gpu()
#        print("GPU mode")
#    else:
    caffe.set_mode_cpu()
    print("CPU mode")
    
    net = caffe.Net(args.model_def, args.pretrained_model, 0)
    ConvLayerL = net.layer_dict['layer_64_1_conv2']
    ConvLayerR = net.layer_dict['layer_64_1_conv1_bypass']
    [N,C,H,W] = ConvLayerL.blobs[0].shape
    ConvLayerL.blobs.add_blob(N)
    ConvLayerL.blobs[1].data[...] += 10
    ConvLayerR.blobs.add_blob(N)
    ConvLayerR.blobs[1].data[...] += 10
    
    ScaleLayer = net.layer_dict['layer_128_1_scale1']
    ScaleLayer.blobs[1].data[...] -= 20

    print("Writing caffemodel file...")
    net.save('bias_'+args.pretrained_model)
    print("Writing txt format caffemodel file...")
    with open('bias_'+args.pretrained_model+"txt",'w') as fid:
        fid.write(str(net))
    


if __name__ == "__main__":
    main(get_args())
