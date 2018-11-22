import numpy as np
import os
import argparse
import caffe
import sys
import pickle
import struct
import cv2

import ParseLFW

def set_gpu(gpuID):
    if gpuID >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpuID)
    else:
        caffe.set_mode_cpu()


class FeatureExtractor(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Net.__init__(self, model_file, caffe.TEST, weights=pretrained_file)
        # self.net = caffe.Net(model_file, caffe.TEST, pretrained_file)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def _get_feature(self, inputs, layer_name):
        input_ = np.zeros((len(inputs), self.image_dims[0], self.image_dims[1], inputs[0].shape[2]), dtype=np.float32)
        #print(input_.shape)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]], dtype=np.float32)
        #print(caffe_in.shape,self.inputs)
        predictions = []
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
            self.blobs[self.inputs[0]].data[...] = caffe_in[ix]
            out = self.forward(end=layer_name)
            #print(ix,np.squeeze(out[layer_name]).shape,np.squeeze(out[layer_name])[0:3])
        #out = self.forward_all(**{self.inputs[0]: caffe_in})
        #print(out)
            outlayer = np.zeros(np.squeeze(out[layer_name]).shape)
            outLayer = np.copy(np.squeeze(out[layer_name]))
            predictions.append(outLayer)
        print("global_pool layer, L:", predictions[0][0:3],"..., R:",predictions[1][0:3])
        #cls_result = []
        #for prediction in predictions:
        #    cls_id = np.argmax(prediction)
        #    cls_result.append(cls_id)
        # print(cls_result)
        return predictions

    def extractFeature(self, input_image, layer_name):
        #inputs = [caffe.io.load_image(input_file)]
        inputs = [input_image/255.]
        self._get_feature(inputs, layer_name)
        #return self.blobs[layer_name].data[0].flatten()
        return self.blobs[layer_name].data[0]
    
    def extractFeatureFromImage(self, input_image, layer_name):
        self._get_feature([input_image], layer_name)
        print(self.blobs[layer_name].data.shape)
        return self.blobs[layer_name].data[0].flatten()
    
    def extractFeatureFromMultiImage(self, inputs, layer_name):
        for idx in range(0,len(inputs)):
            inputs[idx] = inputs[idx]/255.
        features = self._get_feature(inputs, layer_name)
        #features = []
        #print("layer size",features[0].shape,features[1].shape)
        #for i in range(len(inputs)):
        #    features.append(self.blobs[layer_name].data[i].flatten())
        return features

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_lfw_path",
        default='lfw-deepfunneled',
        help="Input LFW data set directory."
    )
    parser.add_argument(
        "--layer_name",
        default='global_pool'
    )
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
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    # parser.add_argument(
        # "--mean_file",
        # default='/home/zhangxin/github/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
        # help="Data set image mean of [Channels x Height x Width] dimensions " +
             # "(numpy array). Set to '' for no mean subtraction."
    # )
    parser.add_argument('--mean', default='0,0,0')
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )

    return parser.parse_args()

def main(args):
    image_dims = [int(s) for s in args.images_dim.split(',')]
    mean, channel_swap = None, None
    #if args.mean_file:
        # mean = np.load(args.mean_file)
    #    mean = np.load(args.mean_file).mean(1).mean(1)
    mean = np.array([int(s) for s in args.mean.split(',')])
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    extractor = FeatureExtractor(args.model_def, args.pretrained_model, image_dims, mean, args.input_scale, args.raw_scale, channel_swap)

    total = 1100
    lfwParser = ParseLFW.ParseLFW(args.input_lfw_path)
    featuredot_match = np.zeros((total,1),np.float)
    featuredot_Unmatch = np.zeros((total,1),np.float)
    featuresL_match = []
    featuresL_Unmatch = []
    featuresR_match = []
    featuresR_Unmatch = []
    imgPair = []
    print("Matched Pairs:")
    for i in range(0,total):
        [matchimgL, matchimgR] = lfwParser.MatchPair_extract()
        #[UnmatchimgL, UnmatchimgR] = lfwParser.MatchPair_extract()
        #matchimgL = cv2.imread("/root/modeltrans_shihl/face_ResNet/image1.jpg")
        #matchimgL = cv2.cvtColor(matchimgL,cv2.COLOR_BGR2RGB)
        #matchimgR = cv2.imread("/root/modeltrans_shihl/face_ResNet/image179414.jpg")
        #matchimgR = cv2.cvtColor(matchimgR,cv2.COLOR_BGR2RGB)
        imgPair = [matchimgL,matchimgR]
        features = extractor.extractFeatureFromMultiImage(imgPair, args.layer_name)
        featureL = features[0]
        featureR = features[1]
        #print(featureL[0:3],featureR[0:3])
        featuresL_match.append(featureL)
        featuredot_match[i,0] = np.dot(featureL,featureR)/(np.linalg.norm(featureL)*np.linalg.norm(featureR))
        featuresR_match.append(featureR)

    #print("Feature Size, Data Type, Data Content\n")
    #print(featureL.shape, type(featureL), featureL,featureR)
    #print("imgL-imgR:",np.max(matchimgL-matchimgR),"max(FeatureL-FeatureR)",np.max(featureL-featureR))
    np.save(args.layer_name + "L_match",featuresL_match)
    np.save(args.layer_name + "R_match",featuresR_match)
    np.save(args.layer_name + "correlation_match",featuredot_match[0:total,0])
    print("Unmatched pairs:\n")
    for i in range(0,total):
        #[matchimgL, matchimgR] = lfwParser.MatchPair_extract()
        [UnmatchimgL, UnmatchimgR] = lfwParser.UnMatchPair_extract()
        #matchimgL = cv2.imread("/root/modeltrans_shihl/face_ResNet/image1.jpg")
        #matchimgL = cv2.cvtColor(matchimgL,cv2.COLOR_BGR2RGB)
        #matchimgR = cv2.imread("/root/modeltrans_shihl/face_ResNet/image179414.jpg")
        #matchimgR = cv2.cvtColor(matchimgR,cv2.COLOR_BGR2RGB)
        imgPair = [UnmatchimgL,UnmatchimgR]
        features = extractor.extractFeatureFromMultiImage(imgPair, args.layer_name)
        featureL = features[0]
        featureR = features[1]
        #print(featureL[0:3],featureR[0:3])
        featuresL_Unmatch.append(featureL)
        featuredot_Unmatch[i,0] = np.dot(featureL,featureR)/(np.linalg.norm(featureL)*np.linalg.norm(featureR))
        featuresR_Unmatch.append(featureR)

    np.save(args.layer_name + "L_Unmatch",featuresL_Unmatch)
    np.save(args.layer_name + "R_Unmatch",featuresR_Unmatch)
    np.save(args.layer_name + "correlation_Unmatch",featuredot_Unmatch[0:total,0])
    
    print("Correlations for matched pairs:",featuredot_match[0:total,0])
    print("Correlations for Unmatched pairs:",featuredot_Unmatch[0:total,0])
    #print("Saved feature to file " + os.getcwd() + "/" + args.layer_name + ".npy")
    #print("Saved feature to file " + os.getcwd() + "/" + args.layer_name + ".npy")


if __name__ == "__main__":
    main(get_args())
