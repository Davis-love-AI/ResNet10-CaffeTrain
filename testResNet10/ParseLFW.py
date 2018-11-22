# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:35:17 2018

test caffe model using lfw dataset
python module to parse LFW dataset
usage:
    import ParseLFW
    a = ParseLFW.ParseLFW("lfw-deepfunneled")
    [L,R] = a.MatchPair_extract()
    [L,R] = a.UnMatchPair_extract()
@author: shihl
"""

import numpy as np
import os
import sys
import argparse
import PIL
from matplotlib import pyplot as plt

#def parse_args():
#    parser = argparse.Argumentparser()
#    
#    parser.add_argument = ("LFW_Path")
#    parser.add_argument(
#        "model_def",
#        help="Model definition file.", type = str
#    )
#    parser.add_argument(
#        "pretrained_model",
#        help="Trained model weights file.", type = str
#    )

class matchPair:
        def __init__(self):
            self.name = 0
            self.fileid_L = 0
            self.fileid_R = 0
            self.filenameL = ""
            self.filenameR = ""
        
class UnmatchPair:
        def __init__(self):
            self.nameL = 0
            self.nameR = 0
            self.fileid_L = 0
            self.fileid_R = 0
            self.filenameL = ""
            self.filenameR = ""
        
class ParseLFW:
    def __init__(self,LFWPath):
        self.matchcount = 0
        print ("Start Reading LFW Data from %s ..." % LFWPath)
        curpwd = os.getcwd()
        if not curpwd.endswith(LFWPath):
            os.chdir(curpwd + "/" + LFWPath)
        self.matchPairs = []
        self.matchidx = 0
        self.UnmatchPairs = []
        self.unmatchidx = 0
        if os.path.exists("pairsDevTrain.txt"):
            fid = open("pairsDevTrain.txt")
            while True:
                fline = fid.readline()
                if len(fline) == 0:
                    break
                #print("pairsDevTrain.txt")
                if(fline.count('\t')==0):
                    self.matchcount = np.int(fline)
                elif(fline.count('\t')==2):
                    matchPair_cur = matchPair();
                    [fline,postfix] = fline.split("\n")
                    [name, idL, idR] = fline.split("\t")
                    matchPair_cur.name = name
                    matchPair_cur.fileid_L = np.int(idL)
                    matchPair_cur.fileid_R = np.int(idR)
                    if np.int(idL) < 10:
                        matchPair_cur.filenameL = LFWPath + "/" + name + "/" + name + "_000" + idL + ".jpg"
                    elif np.int(idL) < 100:
                        matchPair_cur.filenameL = LFWPath + "/" + name + "/" + name + "_00" + idL + ".jpg"
                    else:
                        matchPair_cur.filenameL = LFWPath + "/" + name + "/" + name + "_0" + idL + ".jpg"
                    if np.int(idR) < 10:
                        matchPair_cur.filenameR = LFWPath + "/" + name + "/" + name + "_000" + idR + ".jpg"
                    elif np.int(idR) < 100:
                        matchPair_cur.filenameR = LFWPath + "/" + name + "/" + name + "_00" + idR + ".jpg"
                    else:
                        matchPair_cur.filenameR = LFWPath + "/" + name + "/" + name + "_0" + idR + ".jpg"
                    self.matchPairs.append(matchPair_cur)
                    #print(matchPair_cur.filenameL,matchPair_cur.filenameR)
                    os.chdir(curpwd)
                    #np.save("matchpairs.npy",self.matchPairs)
                elif(fline.count('\t')==3):
                    UnmatchPair_cur = UnmatchPair()
                    [fline,postfix] = fline.split("\n")
                    [nameL, idL, nameR, idR] = fline.split("\t")
                    UnmatchPair_cur.nameL = nameL
                    UnmatchPair_cur.nameR = nameR
                    UnmatchPair_cur.fileid_L = np.int(idL)
                    UnmatchPair_cur.fileid_R = np.int(idR)
                    if np.int(idL) < 10:
                        UnmatchPair_cur.filenameL = LFWPath + "/" + nameL + "/" + nameL + "_000" + idL + ".jpg"
                    elif np.int(idL) < 100:
                        UnmatchPair_cur.filenameL = LFWPath + "/" + nameL + "/" + nameL + "_00" + idL + ".jpg"
                    else:
                        UnmatchPair_cur.filenameL = LFWPath + "/" + nameL + "/" + nameL + "_0" + idL + ".jpg"
                    if np.int(idR) < 10:
                        UnmatchPair_cur.filenameR = LFWPath + "/" + nameR + "/" + nameR + "_000" + idR + ".jpg"
                    elif np.int(idR) < 100:
                        UnmatchPair_cur.filenameR = LFWPath + "/" + nameR + "/" + nameR + "_00" + idR + ".jpg"
                    else:
                        UnmatchPair_cur.filenameR = LFWPath + "/" + nameR + "/" + nameR + "_0" + idR + ".jpg"
                    self.UnmatchPairs.append(UnmatchPair_cur)
                    #print(UnmatchPair_cur.filenameL,UnmatchPair_cur.filenameR)
                    os.chdir(curpwd)
                    #np.save("unmatchpairs.npy",self.UnmatchPairs)
                
            fid.close()
#        if os.path.exists("pairsDevTest.txt"):
#            fid = open("pairsDevTest.txt")
#            while True:
#                fline = fid.readline()
#                if len(fline) == 0:
#                    break
#                #print("pairsDevTest.txt")
#            fid.close()
    def MatchPair_extract(self):
        curpwd = os.getcwd()
#        if not curpwd.endswith(LFWPath):
#            os.chdir(curpwd + "\\" + LFWPath)
        matchpicR = PIL.Image.open(curpwd + "/" + self.matchPairs[self.matchidx].filenameR)
        matchpicL = PIL.Image.open(curpwd + "/" + self.matchPairs[self.matchidx].filenameL)
        print(curpwd, self.matchPairs[self.matchidx].filenameL,self.matchPairs[self.matchidx].filenameR)
        matchimgR = np.asarray(matchpicR)
        matchimgL = np.asarray(matchpicL)
        
        self.matchidx += 1
        #plt.imshow(matchimgL)
        #plt.imshow(matchimgR)
        return matchimgL, matchimgR#notice that the returned image is RGB channel organized
    def UnMatchPair_extract(self):
        curpwd = os.getcwd()
#        if not curpwd.endswith(LFWPath):
#            os.chdir(curpwd + "\\" + LFWPath)
        UnmatchpicR = PIL.Image.open(curpwd + "/" + self.UnmatchPairs[self.unmatchidx].filenameR)
        UnmatchpicL = PIL.Image.open(curpwd + "/" + self.UnmatchPairs[self.unmatchidx].filenameL)
        UnmatchimgR = np.asarray(UnmatchpicR)
        print(curpwd, self.UnmatchPairs[self.unmatchidx].filenameL,self.UnmatchPairs[self.unmatchidx].filenameR)
        UnmatchimgL = np.asarray(UnmatchpicL)
        
        self.unmatchidx += 1

        
        return UnmatchimgL, UnmatchimgR#notice that the returned image is RGB channel organized
