"""
created on Tue Dec 4 2018

train caffe model using XP data
python module to parse XP-face dataset
usage:
    import ParseXPface
    a = ParseXPface.ParseXPface("XPfaceIndexfile")
    cropedface = a.Croppedface_extract(index=-1)
    if index takes the default value, the cropped face will be extracted in a successive way
    else, the cropped face with respect to index is extracted
    ##facelandmarks = a.FaceLandmarks_extract()
@author: shihl
"""

import os
import numpy as np
import visdom
import sys
import PIL


def bbox_from_landmarks(landmarks):
    #landmarks = np.array(list(map(float, landmarks))).reshape(-1, 2)
    landmarks = np.array(landmarks,dtype=np.float)
    min_lx = np.min(landmarks[:, 0])
    min_ly = np.min(landmarks[:, 1])
    max_lx = np.max(landmarks[:, 0])
    max_ly = np.max(landmarks[:, 1])

    size = max(max_lx - min_lx, max_ly - min_ly)

    eye_mid = (landmarks[0] + landmarks[1] + landmarks[2] + landmarks[3]) / 4
    mouth_mid = (landmarks[4] + landmarks[5]) / 2

    h_vector = landmarks[3] - landmarks[0]
    v_vector = mouth_mid - eye_mid

    eye_left = landmarks[0] - h_vector / 4
    eye_right = landmarks[3] + h_vector / 4

    nose_shift = landmarks[6] - (eye_mid + mouth_mid) / 2

    fore_head = eye_mid - v_vector * 0.9
    chin = mouth_mid + v_vector * 0.6

    fix_by_nose = nose_shift
    eye_left = eye_left - fix_by_nose * 1.5
    eye_right = eye_right - fix_by_nose * 1.5
    chin = chin - fix_by_nose

    all_points = np.zeros((13, 2), dtype=np.float32)
    all_points[0: 9] = landmarks[:]
    all_points[9] = fore_head
    all_points[10] = chin
    all_points[11] = eye_left
    all_points[12] = eye_right

    min_x = np.amin(all_points[:, 0])
    min_y = np.amin(all_points[:, 1])

    max_x = np.amax(all_points[:, 0])
    max_y = np.amax(all_points[:, 1])

    min_lx = min_lx - size / 6
    min_ly = min_ly - size / 6
    max_lx = max_lx + size / 6
    max_ly = max_ly + size / 6

    min_x = min(min_x, min_lx)
    min_y = min(min_y, min_ly)

    max_x = max(max_x, max_lx)
    max_y = max(max_y, max_ly)

    return min_x, min_y, max_x, max_y

def RectBbox(min_x, min_y, max_x, max_y, center, faceshape):
    crop_lb_x = np.int(min_x)
    crop_lb_y = np.int(min_y)
    crop_ub_x = np.int(max_x)
    crop_ub_y = np.int(max_y)

    crop_w = crop_ub_x - crop_lb_x
    crop_h = crop_ub_y - crop_lb_y

    #print(crop_w,crop_h)

    if(crop_w > crop_h):
        extd_h = np.int((crop_w - crop_h)/2)
        crop_lb_y -= extd_h
        crop_ub_y += extd_h
    else:
        extd_w = np.int((crop_h - crop_w)/2)
        crop_lb_x -= extd_w
        crop_ub_x += extd_w
    crop_centerx = np.int((crop_lb_x + crop_ub_x)/2)
    crop_centery = np.int((crop_lb_y + crop_ub_y)/2)

    crop_lb_x += center[0] - crop_centerx
    crop_lb_y += center[1] - crop_centery
    crop_ub_x += center[0] - crop_centerx
    crop_ub_y += center[1] - crop_centery

    if(crop_lb_x < 0 or crop_lb_y < 0 or crop_ub_x >= faceshape[0] or crop_ub_y >= faceshape[1]):
        lb_shrink = np.max((0 - crop_lb_x, 0 - crop_lb_y))
        ub_shrink = np.max((crop_ub_x - faceshape[0]+1, crop_ub_y - faceshape[1]+1))
        shrink = np.max((lb_shrink, ub_shrink))
        crop_lb_x += shrink
        crop_lb_y += shrink
        crop_ub_x -= shrink
        crop_ub_y -= shrink
       # print(crop_lb_x,crop_lb_y)
       # print(crop_ub_x,crop_ub_y)

    return crop_lb_x, crop_lb_y, crop_ub_x, crop_ub_y

class  FaceLandmarks:
        def __init__(self):
            #self.faceimage = np.zeros((720, 1080, 3))
            self.imagepath = ""
            self.facelandmark = []
            self.faceidx = 0
            self.faceid = ''
            self.facelabel = 0


class ParseXPface:
    def __init__(self, indexfile):
        print("Start parsing XP face data from %s ....." % indexfile)
        curpwd = os.getcwd()
        [datapath, filename] = indexfile.rsplit('/', 1)
        #self.FaceData = FaceLandmarks()
        self.Faces = []
        self.labeldict = {}
        self.fileidx = 0
        self.facelabel = 0
        if(os.path.exists('ParsedFaceData.npy')):
            self.Faces = np.load('ParsedFaceData.npy')
            return
        with open(indexfile, 'r') as fid:
            while True:
                fline = fid.readline()
                #print(fline,type(fline))
                if len(fline) == 0:
                    break
                line = fline.split(' ')
                imagepath = line[0]
                cords = line[1:]
                self.FaceData = FaceLandmarks()
                [prefix, imagename] = imagepath.rsplit('/', 1)
                [self.FaceData.faceid, surfix] = imagename.split('_', 1)

                if not self.labeldict.has_key(self.FaceData.faceid):
                    self.labeldict[self.FaceData.faceid] = self.facelabel
                    self.FaceData.facelabel = self.facelabel
                    self.facelabel += 1
                    #print("Added face id "+str(self.FaceData.faceid)+':'+str(self.FaceData.facelabel))
                else:
                    self.FaceData.facelabel = self.labeldict[self.FaceData.faceid]


                self.FaceData.faceidx = self.fileidx
                self.FaceData.imagepath = datapath + '/' + imagepath
                #print("Parsing file %s" % imagepath)
                #faceimage = np.asarray(PIL.Image.open(datapath + '/' + imagepath))
                self.FaceData.facelandmark.append(np.asarray(cords[0:18], dtype=np.int))
                self.fileidx += 1
                self.Faces.append(self.FaceData)
            print("Parsed %d files" % self.fileidx)
        self.fileidx = 0
        np.save('ParsedFaceImageIndex.npy', self.Faces)

        if not os.path.exists("trainxpfaceidx.txt"):
            flog = open("trainxpfaceidx.txt","w")
            for i in range(0,len(self.Faces)):
                flog.write("file:" + self.Faces[i].imagepath + "face: "+self.Faces[i].faceid + " faceLabel: " + str(self.Faces[i].facelabel) + " fileidx: " \
                        + str(self.Faces[i].faceidx) +"\n")
            flog.close()

    def Facelandmarks_extract(self, index = -1):
        if index == -1:
            return self.Faces
        else:
            return [self.Faces[index]]

    def Croppedface(self, index=-1, drawldmk = 0, viz = 0):
        if index == -1:
            if viz:
              viz = visdom.Visdom(port=10031, env='XPFaces')
            faceimage = np.copy(np.asarray(PIL.Image.open(self.Faces[self.fileidx].imagepath)))
            #print(self.Faces[self.fileidx].faceimage.shape)
            if drawldmk:
                vizimage = (faceimage.transpose(2, 0, 1))
            else:
                vizimage = np.copy(faceimage.transpose(2, 0, 1))
            vizimagefile = self.Faces[self.fileidx].imagepath

            landmarks = self.Faces[self.fileidx].facelandmark[0]
            landmarks = landmarks.reshape(-1,2)

            vizimage[:,landmarks[0:4,1],landmarks[0:4,0]] = 255
            vizimage[0,landmarks[4:6,1],landmarks[4:6,0]] = 255
            vizimage[1:3,landmarks[6,1],landmarks[6,0]] = 255
            vizimage[1,landmarks[7:9,1],landmarks[7:9,0]] = 255

            [min_x,min_y,max_x,max_y] = bbox_from_landmarks(landmarks)

            nose_center = np.copy(landmarks[6])

            [face_h, face_w, n] = faceimage.shape
            [crop_lb_x, crop_lb_y, crop_ub_x, crop_ub_y] = RectBbox(min_x, min_y, max_x, max_y, nose_center, [face_w, face_h])

            vizimage[:,np.int(crop_lb_y),np.int(crop_lb_x):np.int(crop_ub_x)] = 255
            vizimage[:,np.int(crop_ub_y),np.int(crop_lb_x):np.int(crop_ub_x)] = 255
            vizimage[:,np.int(crop_lb_y):np.int(crop_ub_y),np.int(crop_ub_x)] = 255
            vizimage[:,np.int(crop_lb_y):np.int(crop_ub_y),np.int(crop_lb_x)] = 255


            if viz:
              viz.image(vizimage, win='XPFaces', opts=dict(title=vizimagefile))
            #saveimage = faceimage[np.int(min_y):np.int(max_y),np.int(min_x):np.int(max_x),:]
            saveimage = faceimage[crop_lb_y:crop_ub_y,crop_lb_x:crop_ub_x,0:3]

            if (crop_lb_y < 0 or crop_ub_y > face_h or crop_lb_x < 0 or crop_ub_x > face_w):
                return saveimage, -1

            pilimage = PIL.Image.fromarray(saveimage)
            #pilimage = pilimage.resize((224,224))
            saveimage = np.asarray(pilimage.resize((224, 224)))
            if viz:
              viz.image(saveimage.transpose(2,0,1),win='XPCropped Faces',opts = dict(title=vizimagefile))
              #print(vizimagefile, self.Faces[self.fileidx].faceid, self.fileidx, saveimage.shape)
            #pilimage.save(imagefile)
            self.fileidx += 1
            if(self.fileidx == len(self.Faces)):
              self.fileidx = 0
            return saveimage, self.Faces[self.fileidx-1].facelabel
        else:
            if viz:
              viz = visdom.Visdom(port=10031, env='XPFaces')
            faceimage = np.copy(np.asarray(PIL.Image.open(self.Faces[index].imagepath)))
            #print(faceimage.shape)
            landmarks = self.Faces[index].facelandmark[0]
            landmarks = landmarks.reshape(-1,2)
            if drawldmk:
                vizimage = (faceimage.transpose(2, 0, 1))
            else:
                vizimage = np.copy(faceimage.transpose(2, 0, 1))

            vizimagefile = self.Faces[index].imagepath
            vizimage[:,landmarks[0:4,1],landmarks[0:4,0]] = 255
            vizimage[0,landmarks[4:6,1],landmarks[4:6,0]] = 255
            vizimage[1,landmarks[6:9,1],landmarks[6:9,0]] = 255

            [min_x,min_y,max_x,max_y] = bbox_from_landmarks(landmarks)
            nose_center = np.copy(landmarks[6])

            [face_h, face_w, n] = faceimage.shape
            [crop_lb_x, crop_lb_y, crop_ub_x, crop_ub_y] = RectBbox(min_x, min_y, max_x, max_y, nose_center, [face_w, face_h])

            vizimage[:,np.int(crop_lb_y),np.int(crop_lb_x):np.int(crop_ub_x)] = 255
            vizimage[:,np.int(crop_ub_y),np.int(crop_lb_x):np.int(crop_ub_x)] = 255
            vizimage[:,np.int(crop_lb_y):np.int(crop_ub_y),np.int(crop_ub_x)] = 255
            vizimage[:,np.int(crop_lb_y):np.int(crop_ub_y),np.int(crop_lb_x)] = 255

            if viz:
              viz.image(vizimage, win='XPFaces', opts=dict(title=vizimagefile))

            saveimage = faceimage[crop_lb_y:crop_ub_y,crop_lb_x:crop_ub_x,0:3]

            if (crop_lb_y < 0 or crop_ub_y > face_h or crop_lb_x < 0 or crop_ub_x > face_w):
            #    print("lb", crop_lb_x,crop_lb_y)
            #    print("ub", crop_ub_x,crop_ub_y)
            #    print("h, w", face_h, face_w)
                return saveimage, -1
            #print(min_x,min_y,max_x,max_y,nose_center,crop_lb_y,crop_ub_y,crop_lb_x,crop_ub_x,saveimage.shape)

            pilimage = PIL.Image.fromarray(saveimage)
            saveimage = np.asarray(pilimage.resize((224, 224)))
            if viz:
              viz.image(saveimage.transpose(2,0,1),win='XPCropped Faces',opts = dict(title=vizimagefile))
              print(vizimagefile, self.Faces[index].faceid, self.Faces[index].facelabel, index, saveimage.shape)
            return saveimage, self.Faces[index].facelabel


