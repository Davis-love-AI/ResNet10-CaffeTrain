import sys
import os
print(sys.path)
print(sys.version)
import cv2
import caffe
import numpy as np
import cv2
import base64
import random
#import _pickle as pickle
from operator import itemgetter

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
def convertImage(data):
    imgdata = base64.b64decode(data)
    img = cv2.imdecode(np.frombuffer(imgdata,dtype=np.uint8),-1)
    [h,w,n] = img.shape
    if h >= w:
        img_resize = img[np.int((h-w)/2):h-np.int((h-w)/2),:,:]
    elif w > h:
        img_resize = img[:,np.int((w-h)/2):w-np.int((w-h)/2),:]
    img_resize = cv2.resize(img,(112,112))
    return img_resize

################################################################################
#########################Data Layer By Python###################################
################################################################################
class input_layer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class train_input_layer(input_layer):
    def setup(self, bottom, top):
        self.batch_size = 32
        image_size = 112

        self.tsv_loader = \
        TsvLoader('/data/public/face/recognition/msceleb/raw/TrainData_Base.tsv')

        top[0].reshape(self.batch_size, 3, image_size, image_size)#very important size specification!
        top[1].reshape(self.batch_size, 1)

    def forward(self, bottom, top):

        for itt in range(self.batch_size):
            im, label= self.tsv_loader.load_next_image()#opencv standard image, size(h w n)
            top[0].data[itt, ...] = im.transpose(2,0,1)
            top[1].data[itt, ...] = label

class test_input_layer(input_layer):
    def setup(self, bottom, top):
        self.batch_size = 10
        image_size = 112

        self.tsv_loader = \
        TsvLoader_for_test('/data/public/face/recognition/msceleb/raw/TrainData_Base.tsv')

        top[0].reshape(self.batch_size, 3, image_size, image_size)
        top[1].reshape(self.batch_size, 1)

    def forward(self, bottom, top):

        for itt in range(self.batch_size):
            im, label= self.tsv_loader.load_next_image()
            top[0].data[itt, ...] = im.transpose(2,0,1)
            top[1].data[itt, ...] = label


class ITEM:
    def __init__(self):
        self.fp = 0
        self.MID = ""
        self.faceindex = 0
        self.fileidx = 0

class TsvLoader(object):
    def __init__(self, tsv_file_path):

        print ("Start Reading msCeleb tsv Data %s...",tsv_file_path)

        with open(tsv_file_path,'r') as fid1:
            self.fp = fid1.tell()
            self.tsv_file_path = tsv_file_path
            self.faceindex = 0
            self.freeBaseMID = []
            self.labeldict = {}
            self.fileidx = 0
            self.fileitem = []

            #item = ITEM()
            while True:
                item = ITEM()
                item.fp = fid1.tell()
                fline = fid1.readline()
                if(len(fline)==0 or self.faceindex > 10000):
                    break
                if(fline.count('\t')==4):
                    [imageID_cur,FaceData_cur,FreeBaseMID_cur,ImgSearchRk_cur,ImageURL_cur] = fline.split("\t")             
                else:
                    [FreeBaseMID_cur, ImgSearchRk_cur, ImageURL_cur, PageURL_cur, imageID_cur,\
                    FaceRect_cur, FaceData_cur] = fline.split("\t")
                item.MID = FreeBaseMID_cur
                item.fileidx = self.fileidx
                if not self.labeldict.has_key(FreeBaseMID_cur):
                    #self.faceindex += 1
                    #item.faceindex = np.copy(self.faceindex)
                    #item.fp = fid1.tell()
                    #item.MID = FreeBaseMID_cur
                    #self.fileitem.append(item)
                    self.labeldict[FreeBaseMID_cur] = self.faceindex
                    #print("Caching image ID:",FreeBaseMID_cur,"label:",self.faceindex,"file idx:",item.fileidx,"fp:",item.fp)
                    self.faceindex += 1
                item.faceindex = self.faceindex - 1
                self.fileidx += 1
                if self.faceindex > 10000:
                    break
                self.fileitem.append(item)

        
        np.random.shuffle(self.fileitem)
        np.save("testEntry.npy",self.fileitem[0:100000])
        del(self.fileitem[0:100000])
        self.readfileidx = 0
        if not os.path.exists("trainfaceidx.txt"):
            flog = open("trainfaceidx.txt","w")
            for i in range(0,len(self.fileitem)):
                flog.write("MID: "+self.fileitem[i].MID + " faceLabel: " + str(self.fileitem[i].faceindex) + "+" + str(self.labeldict[self.fileitem[i].MID]) + " fileidx: " \
                        + str(self.fileitem[i].fileidx) + " fp: " + str(self.fileitem[i].fp)+"\n")
                if self.fileitem[i].faceindex != self.labeldict[self.fileitem[i].MID]:
                    flog.write("Inequality!\n")
            flog.close()
        #np.random.shuffle(self.fileitem)
        #np.save("testEntry.npy",self.fileitem[0:100000])
        #del(self.fileitem[0:100000])
        print (" Tsv Data file for training cached ready, totally "+str(len(self.labeldict))+"different faces," + "and " + str(len(self.fileitem)) + " images!\n")


    def load_next_image(self):
        if(self.readfileidx == 0):
            #randidx = random.randrange(len(self.fileitem))
            print("Begining of a new epoch, randomly shffuling input set...\n")
            np.random.shuffle(self.fileitem)
        with open(self.tsv_file_path,'r') as fid1:
            fid1.seek(self.fileitem[self.readfileidx].fp,0)
        # read tsv data and extract the image

            fline = fid1.readline()
            if len(fline)==0:
                print("EOF!\n")
                return
            if(fline.count('\t')==4):
                [imageID_cur,FaceData_cur,FreeBaseMID_cur,ImgSearchRk_cur,ImageURL_cur] = fline.split("\t")
            else:
                [FreeBaseMID_cur, ImgSearchRk_cur, ImageURL_cur, PageURL_cur, imageID_cur,\
                    FaceRect_cur, FaceData_cur] = fline.split("\t")
            im = convertImage(FaceData_cur)
            #if not self.labeldict.has_key(FreeBaseMID_cur):
                #self.index = self.index + 1
                #self.labeldict[FreeBaseMID_cur] = self.index
            
            #if np.mod(self.readfileidx,100) == 0:
            if self.fileitem[self.readfileidx].faceindex != self.labeldict[FreeBaseMID_cur]:
                print("Iter", str(self.readfileidx), "Loading image ID:",FreeBaseMID_cur," label: ",self.fileitem[self.readfileidx].faceindex, "+", self.labeldict[FreeBaseMID_cur],\
                    " file idx:",self.fileitem[self.readfileidx].fileidx," fp:",self.fileitem[self.readfileidx].fp)

            label = self.labeldict[FreeBaseMID_cur]
            #if label != self.labeldict[FreeBaseMID_cur]:
                #print("Label Error!\n")
            #self.fp = fid1.tell()
            self.readfileidx += 1
            if self.readfileidx == len(self.fileitem):
                self.readfileidx = 0
            #randidx = random.randrange(len(self.fileitem))
            #print("Loading image:",FreeBaseMID_cur,"label:",label)
            
        return im, label



########################

class TsvLoader_for_test(object):
    def __init__(self, tsv_file_path):

        print ("Start Reading msCeleb tsv Data %s...",tsv_file_path)

        with open(tsv_file_path,'r') as fid1:
            self.fp = fid1.tell()
            self.tsv_file_path = tsv_file_path
            self.faceindex = 0
            self.freeBaseMID = []
            self.labeldict = {}
            self.fileidx = 0
            self.fileitem = []

            #item = ITEM()
            while True:
                item = ITEM()
                item.fp = fid1.tell()
                fline = fid1.readline()
                if(len(fline)==0 or self.faceindex > 10000):
                    break
                if(fline.count('\t')==4):
                    [imageID_cur,FaceData_cur,FreeBaseMID_cur,ImgSearchRk_cur,ImageURL_cur] = fline.split("\t")             
                else:
                    [FreeBaseMID_cur, ImgSearchRk_cur, ImageURL_cur, PageURL_cur, imageID_cur,\
                    FaceRect_cur, FaceData_cur] = fline.split("\t")
                item.MID = FreeBaseMID_cur
                item.fileidx = self.fileidx
                if not self.labeldict.has_key(FreeBaseMID_cur):
                    #self.faceindex += 1
                    #item.faceindex = np.copy(self.faceindex)
                    #item.fp = fid1.tell()
                    #item.MID = FreeBaseMID_cur
                    #self.fileitem.append(item)
                    self.labeldict[FreeBaseMID_cur] = self.faceindex
                    #print("Caching image ID:",FreeBaseMID_cur,"label:",self.faceindex,"file idx:",item.fileidx,"fp:",item.fp)
                    self.faceindex += 1
                item.faceindex = self.faceindex - 1
                self.fileidx += 1
                if self.faceindex > 10000:
                    break
                self.fileitem.append(item)

        
        np.random.shuffle(self.fileitem)
        if os.path.exists("testEntry.npy"):
            self.fileitem = np.load("testEntry.npy")
        else:
            self.fileitem = self.fileitem[0:100000]

        self.readfileidx = 0
        if not os.path.exists("testfaceidx.txt"):
            flog = open("testfaceidx.txt","w")
            for i in range(0,len(self.fileitem)):
                flog.write("MID: "+self.fileitem[i].MID + " faceLabel: " + str(self.fileitem[i].faceindex) + "+" + str(self.labeldict[self.fileitem[i].MID]) + " fileidx: " \
                        + str(self.fileitem[i].fileidx) + " fp: " + str(self.fileitem[i].fp)+"\n")
                if self.fileitem[i].faceindex != self.labeldict[self.fileitem[i].MID]:
                    flog.write("Inequality!\n")
            flog.close()
                
        print (" Tsv Data file for testing cached ready, totally %d images!\n" % len(self.fileitem))


    def load_next_image(self):
        #if(self.readfileidx == 0):
        randidx = random.randrange(len(self.fileitem))
            #np.random_shuffle(self.fileitem)
        with open(self.tsv_file_path,'r') as fid1:
            fid1.seek(self.fileitem[randidx].fp,0)
        # read tsv data and extract the image

            fline = fid1.readline()
            if len(fline)==0:
                print("EOF!\n")
                return
            if(fline.count('\t')==4):
                [imageID_cur,FaceData_cur,FreeBaseMID_cur,ImgSearchRk_cur,ImageURL_cur] = fline.split("\t")
            else:
                [FreeBaseMID_cur, ImgSearchRk_cur, ImageURL_cur, PageURL_cur, imageID_cur,\
                    FaceRect_cur, FaceData_cur] = fline.split("\t")
            im = convertImage(FaceData_cur)
            #if not self.labeldict.has_key(FreeBaseMID_cur):
                #self.index = self.index + 1
                #self.labeldict[FreeBaseMID_cur] = self.index
            
            #if np.mod(self.readfileidx,100) == 0:
            #if self.fileitem[randidx].faceindex != self.labeldict[FreeBaseMID_cur]:
            #    print("Iter", str(self.readfileidx), "randidx:", randidx, "Loading image ID:",FreeBaseMID_cur," label: ",self.fileitem[randidx].faceindex, "+", self.labeldict[FreeBaseMID_cur],\
            #        " file idx:",self.fileitem[randidx].fileidx," fp:",self.fileitem[randidx].fp)

            label = self.labeldict[FreeBaseMID_cur]
            #if label != self.labeldict[FreeBaseMID_cur]:
                #print("Label Error!\n")
            #self.fp = fid1.tell()
            self.readfileidx += 1
            if self.readfileidx == len(self.fileitem):
                self.readfileidx = 0
            #randidx = random.randrange(len(self.fileitem))
            #print("Loading image:",FreeBaseMID_cur,"label:",label)
            
        return im, label
