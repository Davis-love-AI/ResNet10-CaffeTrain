"""reated on Thu Nov 15 11:11:18 2018

@author: shihl

match the train and test phase loss and accuracy
"""
import sys
import io
import re
import argparse
from matplotlib import pyplot as plt
import numpy as np
import visdom

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",help="Input caffe train log file name",type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    viz = visdom.Visdom(port=10031,env=("Parsed caffe training and testing result from "+ args.filename))
    filename = args.filename
    restr_base_lr = r'base_lr: ([0-9].[0-9]+)'
    restr_weight_decay = r'weight_decay: ([0-9].[0-9]+)'
    restr_train = r'Train net output #0: loss = ([0-9].[0-9]+)'
    restr_test = r'Test net output #1: loss = ([0-9].[0-9]+)'
    restr_testaccu = r'Test net output #0: accuracy = ([0-9].[0-9]+)'
    restr_trainiter = r'Iteration ([0-9]+), lr'
    restr_testiter = r'Iteration ([0-9]+), Testing net'
    fid = open(filename,'r')
    line = 0
    Train_loss = np.array([])
    Train_iter = np.array([])
    Test_loss = np.array([])
    Test_accu = np.array([])
    Test_iter = np.array([])
    base_lr = 0
    weight_decay = 0
    for fileline in fid:
    #filecontent = fid.read()
        filecontent = fileline.rstrip()
        if(re.search(restr_train,filecontent)):
            #print(line, re.search(restr,filecontent).group(1))
            Train_loss = np.append(Train_loss, np.float(re.search(restr_train,filecontent).group(1)))
        if(re.search(restr_trainiter,filecontent)):
            Train_iter = np.append(Train_iter,np.float(re.search(restr_trainiter,filecontent).group(1)))
        if(re.search(restr_test,filecontent)):
            #print(line, re.search(restr,filecontent).group(1))
            Test_loss = np.append(Test_loss, np.float(re.search(restr_test,filecontent).group(1)))
        if(re.search(restr_testaccu,filecontent)):
            #print(line, re.search(restr,filecontent).group(1))
            Test_accu = np.append(Test_accu,np.float(re.search(restr_testaccu,filecontent).group(1)))
        if(re.search(restr_testiter,filecontent)):
            Test_iter = np.append(Test_iter,np.float(re.search(restr_testiter,filecontent).group(1)))
        if(re.search(restr_base_lr,filecontent)):
            base_lr = np.float(re.search(restr_base_lr,filecontent).group(1))
        if(re.search(restr_weight_decay,filecontent)):
            weight_decay = np.float(re.search(restr_weight_decay,filecontent).group(1))

        line += 1
    #print(filecontent)
    fid.close()
    iteration_epoch = 17910
    #print(re.match(r'Train','Train net output'))
    test_len = np.min((len(Test_accu),len(Test_iter),len(Test_loss)))
    #plt.figure(1)
    #plt.subplot(211)
    #viz.text("Parsed caffe training and testing result from "+ filename)
    train_len = np.min((len(Train_iter),len(Train_loss)))
    #plt.plot(Test_iter[0:test_len]/15563,Test_loss[0:test_len],'-',linewidth = 2.0)
    #plt.xlabel('Epoches')
    #plt.ylabel('Test Loss')
    #plt.grid()
    viz.line(X=Test_iter[0:test_len]/iteration_epoch, Y=Test_loss[0:test_len],win='Test Loss',opts=dict(title=('Test Loss,\n base_lr:'+str(base_lr)+'weight_decay:'+str(weight_decay))))
    #viz.matplot(plt)
    #plt.subplot(212)
    #plt.plot(Train_iter[0:train_len]/15563,Train_loss[0:train_len],'-')
    #plt.xlabel('Epoches')
    #plt.ylabel('Train Loss')
    #plt.grid()
    #viz.matplot(plt)
    viz.line(X=Train_iter[0:train_len]/iteration_epoch, Y=Train_loss[0:train_len],win='Train Loss',opts=dict(title=('Train Loss, base_lr:'+str(base_lr)+'weight_decay:'+str(weight_decay))))
    #viz.matplot(plt)
    #plt.figure(2)
    #plt.plot(Test_iter[0:test_len]/15563,Test_accu[0:test_len])
    #plt.xlabel('Epoches')
    #plt.ylabel('Test Accuracy')
    #plt.grid()
    #viz.matplot(plt)
    viz.line(X=Test_iter[0:test_len]/iteration_epoch, Y=Test_accu[0:test_len],win='Test Accuracy',opts=dict(title=('Test Accuracy, base_lr:'+str(base_lr)+'weight_decay:'+str(weight_decay))))
    #viz.matplot(plt)
    #plt.show()
