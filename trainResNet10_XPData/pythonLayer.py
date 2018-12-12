import sys
import os
print(sys.path)
print(sys.version)
sys.path.insert(0,'/root/caffe-master/python')
import cv2
import caffe
import numpy as np
import cv2
import base64
import random
import ParseXPface
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
    img_resize = cv2.resize(img,(224,224))
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
        image_size = 224

        self.Image_loader = \
        ImageLoader('/data/public/face/recognition/msceleb/raw/TrainData_Base.tsv','/data-face/xp_data/landmarks.txt')

        top[0].reshape(self.batch_size, 3, image_size, image_size)#very important size specification!
        top[1].reshape(self.batch_size, 1)

    def forward(self, bottom, top):

        for itt in range(self.batch_size):
            im, label= self.Image_loader.load_next_image()#opencv standard image, size(h w n)
            top[0].data[itt, ...] = im.transpose(2,0,1)
            top[1].data[itt, ...] = label

class test_input_layer(input_layer):
    def setup(self, bottom, top):
        self.batch_size = 10
        image_size = 224

        self.Image_loader = \
        ImageLoader_for_test('/data/public/face/recognition/msceleb/raw/TrainData_Base.tsv','/data-face/xp_data/landmarks.txt')

        top[0].reshape(self.batch_size, 3, image_size, image_size)
        top[1].reshape(self.batch_size, 1)

    def forward(self, bottom, top):

        for itt in range(self.batch_size):
            im, label= self.Image_loader.load_next_image()
            top[0].data[itt, ...] = im.transpose(2,0,1)
            top[1].data[itt, ...] = label


class ITEM:
    def __init__(self):
        self.fp = 0
        self.MID = ""
        self.faceindex = 0
        self.fileidx = 0
        #self.labeldict = {}

class ImageLoader(object):
    def __init__(self, tsv_file_path, xp_face_path):

        print ("Start Reading msCeleb tsv Data %s...",tsv_file_path)

        with open(tsv_file_path,'r') as fid1:
            self.fp = fid1.tell()
            self.tsv_file_path = tsv_file_path
            self.faceindex = 0
            self.freeBaseMID = []
            self.labeldict = {}
            self.fileidx = 0
            self.fileitem = []
            logflag = 0

            #item = ITEM()
            while True:
                item = ITEM()
                item.fp = fid1.tell()
                fline = fid1.readline()
                if(len(fline)==0 or self.faceindex > 10000 - 415):
                    break
                if(fline.count('\t')==4):
                    [imageID_cur,FaceData_cur,FreeBaseMID_cur,ImgSearchRk_cur,ImageURL_cur] = fline.split("\t")
                else:
                    [FreeBaseMID_cur, ImgSearchRk_cur, ImageURL_cur, PageURL_cur, imageID_cur,\
                    FaceRect_cur, FaceData_cur] = fline.split("\t")
                item.MID = FreeBaseMID_cur
                item.fileidx = self.fileidx
                if not self.labeldict.has_key(FreeBaseMID_cur):
                    self.labeldict[FreeBaseMID_cur] = self.faceindex
                    #print("Caching image ID:",FreeBaseMID_cur,"label:",self.faceindex,"file idx:",item.fileidx,"fp:",item.fp)
                    item.faceindex = self.faceindex
                    self.faceindex += 1
                else:
                    item.faceindex = self.labeldict[FreeBaseMID_cur]
                self.fileidx += 1
                if self.faceindex > 10000 - 415:
                    self.labeldict.pop(FreeBaseMID_cur)
                    break
                self.fileitem.append(item)

        if not os.path.exists("trainfaceidx.txt"):
            flog = open("trainfaceidx.txt","w")
            for i in range(0,len(self.fileitem)):
                flog.write("MID: "+self.fileitem[i].MID + " faceLabel: " + str(self.fileitem[i].faceindex) + "+" + str(self.labeldict[self.fileitem[i].MID]) + " fileidx: " \
                        + str(self.fileitem[i].fileidx) + " fp: " + str(self.fileitem[i].fp)+"\n")
                if self.fileitem[i].faceindex != self.labeldict[self.fileitem[i].MID]:
                    flog.write("Inequality!\n")
            flog.close()
            logflag = 1
        print (" Tsv Data file for training cached ready, totally "+str(len(self.labeldict))+"different faces," + "and " + str(len(self.fileitem)) + " images!\n")


        print("Start parsing XP face data %s..." % xp_face_path)
        self.XPface = ParseXPface.ParseXPface(xp_face_path)
        XPfacelist = self.XPface.Facelandmarks_extract()
        #self.fileitem.extend(XPfacelist)

        if os.path.exists("trainfaceidx.txt") and logflag:
            flog = open("trainfaceidx.txt","a+")
            for i in range(0,len(XPfacelist)):
                XPfacelist[i].facelabel += np.int(len(self.labeldict))
                flog.write("MID: "+XPfacelist[i].faceid + " faceLabel: " + str(XPfacelist[i].facelabel) \
                      + ', faceid'  + str(XPfacelist[i].faceidx) + ', fileid' + str(i+len(self.fileitem))  + " filepath: " + str(XPfacelist[i].imagepath)+"\n")
            flog.close()
        self.fileitem.extend(XPfacelist)
        np.save("TrainEntry.npy",[self.fileitem, self.labeldict])
        np.random.shuffle(self.fileitem)
        np.save("testEntry.npy", [self.fileitem[0:100000], self.labeldict])
        del(self.fileitem[0:100000])
        self.readfileidx = 0

    def load_next_image(self):
        if(self.readfileidx == 0):
            #randidx = random.randrange(len(self.fileitem))
            print("Begining of a new epoch, randomly shffuling input set...\n")
            np.random.shuffle(self.fileitem)
        if(self.fileitem[self.readfileidx].__module__ != "ParseXPface"):
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
        else:
            #print('XPface')
            [im, label] = self.XPface.Croppedface(self.fileitem[self.readfileidx].faceidx)
            label = self.fileitem[self.readfileidx].facelabel
            self.readfileidx += 1
            if self.readfileidx == len(self.fileitem):
                self.readfileidx = 0
        return im, label



########################

class ImageLoader_for_test(object):
    def __init__(self, tsv_file_path, xp_face_path):
        self.labeldict = {}
        print ("Start Reading test Data %s...",tsv_file_path)
        if os.path.exists("testEntry.npy"):
            self.tsv_file_path = tsv_file_path
            [self.fileitem, self.labeldict] = np.load("testEntry.npy")
        else:
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
            self.fileitem = self.fileitem[0:100000]

        self.readfileidx = 0
        #if not os.path.exists("testfaceidx.txt"):
        #    flog = open("testfaceidx.txt","w")
        #    for i in range(0,len(self.fileitem)):
        #        flog.write("MID: "+self.fileitem[i].MID + " faceLabel: " + str(self.fileitem[i].faceindex) + "+" + str(self.labeldict[self.fileitem[i].MID]) + " fileidx: " \
        #                + str(self.fileitem[i].fileidx) + " fp: " + str(self.fileitem[i].fp)+"\n")
        #        if self.fileitem[i].faceindex != self.labeldict[self.fileitem[i].MID]:
        #            flog.write("Inequality!\n")
        #    flog.close()
        print("Start parsing XP face data %s..." % xp_face_path)
        self.XPface = ParseXPface.ParseXPface(xp_face_path)
        XPfacelist = self.XPface.Facelandmarks_extract()

        print (" Data for testing is ready, totally %d images!\n" % len(self.fileitem))


    def load_next_image(self):
        if(self.readfileidx == 0):
        #randidx = random.randrange(len(self.fileitem))
            np.random.shuffle(self.fileitem)
        if(self.fileitem[self.readfileidx].__module__ != "ParseXPface"):
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
              label = self.labeldict[FreeBaseMID_cur]
              self.readfileidx += 1
              if self.readfileidx == len(self.fileitem):
                  self.readfileidx = 0
        else:
          im, label = self.XPface.Croppedface(self.fileitem[self.readfileidx].faceidx)
          label = self.fileitem[self.readfileidx].facelabel
          self.readfileidx += 1
          if self.readfileidx == len(self.fileitem):
            self.readfileidx = 0

        return im, label
