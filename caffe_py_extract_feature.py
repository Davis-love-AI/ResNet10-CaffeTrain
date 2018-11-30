#encoding=utf8
'''
python2.7
'''
import numpy as np
import os
import argparse
import caffe
import sys
import pickle
import struct
import sys,cv2
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')


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
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]], dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[ix], in_)
        # out = self.forward()
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]]
        cls_result = []
        for prediction in predictions:
            cls_id = np.argmax(prediction)
            cls_result.append(cls_id)
        # print(cls_result)
        return cls_result
    
    def _network_forward(self,inputs):
        input_ = np.zeros((len(inputs), self.image_dims[0], self.image_dims[1], inputs[0].shape[2]), dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]], dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[ix],in_)

        self.forward_all(**{self.inputs[0]: caffe_in})

    # 提取特征并保存为相应地文件
    def extractFeature(self, input_file, layer_name):
        inputs = [caffe.io.load_image(input_file)]
        self._get_feature(inputs, layer_name)
        #return self.blobs[layer_name].data[0].flatten()
        return self.blobs[layer_name].data[0]
    def extractallFeature(self, input_file):
        inputs = [caffe.io.load_image(input_file)]
        self._network_forward(inputs)
        features = {}
        for blob_name in self.blobs.keys():
            if "split" in blob_name:
                continue
            blobdata = np.copy(self.blobs[blob_name].data[0])
            print("Processing blob: "+ blob_name + "Dim: " + str(blobdata.shape))
            features[blob_name] = blobdata
        return features
    
        # 提取特征并保存为相应地文件
    def extractFeatureFromImage(self, input_image, layer_name):
        self._get_feature([input_image], layer_name)
        return self.blobs[layer_name].data[0].flatten()
    
    def extractFeatureFromMultiImage(self, inputs, layer_name):
        self._get_feature(inputs, layer_name)
        features = []
        for i in range(len(inputs)):
            features.append(self.blobs[layer_name].data[i].flatten())
        return features


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default='/home/zhangxin/pic/2.png',
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "--layer_name",
        default='pool5/7x7_s1'
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
	"--gpuid",
	default='0',
	type=int,
	help="The ID of GPU to run."
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
	caffe.set_device(args.gpuid)
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    extractor = FeatureExtractor(args.model_def, args.pretrained_model, image_dims, mean, args.input_scale, args.raw_scale, channel_swap)

    if args.layer_name != "all":
        feature = extractor.extractFeature(args.input_file, args.layer_name)
        print("Feature Size, Data Type, Data Content\n")
        print(feature.shape, type(feature), feature)
        np.save(args.layer_name + "_reluadd",feature)
        print("Saved layer feature to file " + os.getcwd() + "/" + args.layer_name + "_reluadd.npy")
    else:
        features = extractor.extractallFeature(args.input_file)
        for name, feature in features.items():
            np.save(name + "_reluadd_all", feature)

if __name__ == "__main__":
    main(get_args())
    
