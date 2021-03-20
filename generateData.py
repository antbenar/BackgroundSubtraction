import numpy as np
import tensorflow as tf
import random as rn
import os, sys
import os
import glob
from keras.preprocessing import image as kImage
from sklearn.utils import compute_class_weight

# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

class GenerateData:
    def __init__(self, maxW, maxH):
        self.maxW = maxW
        self.maxH = maxH
    
    def generateTemporalInformation(self, X_list, framesBack):
        Xresult = []
        
        for i in range(len(X_list) - framesBack):
            Xtemp = []
            for j in range(i, i + framesBack):
                Xtemp.append(X_list[j])
            Xtemp = np.asarray(Xtemp)
            Xresult.append(Xtemp)
            #print(str(i) + ' -> ' + str(Xtemp.shape))
            
        Xresult = np.asarray(Xresult)
        return  Xresult
    
    def generateFiveTimes(self, X_list, framesBack):
        Xresult = []
        
        for i in range(framesBack - 1, len(X_list)):
            Xtemp = []
            for j in range(framesBack):
                Xtemp.append(X_list[i])
            Xtemp = np.asarray(Xtemp)
            Xresult.append(Xtemp)
            #print(str(i) + ' -> ' + str(Xtemp.shape))
            
        Xresult = np.asarray(Xresult)
        return  Xresult
    
    """
    def make_5d(data, n_look_back):
        #~ print('Reshaping data as 5D...\n')
        #~ n_look_back = 2
        k = 0
        n_samples = len(data)
        tmp = []
        data_5d = np.empty((n_samples-(n_look_back-1), n_look_back, data.shape[1], data.shape[2], data.shape[3]), dtype='float32')
        
        for i in range(0, n_samples):
            tmp = data[i:i+n_look_back]
            if tmp.shape[0] == n_look_back:
                for rotate_channel_id in range(0, n_look_back): # rotate the channels such that bring the current input as first channel
                    tmp[rotate_channel_id] = tmp[n_look_back-1-rotate_channel_id]
                #tmp = tmp.reshape(1, n_look_back, data.shape[1], data.shape[2], data.shape[3])
    
                tmpArray = []
                for i in range(0, n_look_back): # rotate the channels such that bring the current input as first channel
                    aux = np.asarray(tmp[i])
                    tmpArray = np.concatenate((tmpArray,aux),axis = 0)
    
                data_5d[k] = tmpArray
                tmp = [] # clear tmp
                k = k + 1
    
        return data_5d
    
    
    """
    def make_5d(self, data, n_look_back):
        #~ print('Reshaping data as 5D...\n')
        #~ n_look_back = 2
        k = 0
        n_samples = len(data)
        tmp = []
        data_5d = np.empty((n_samples-(n_look_back-1), n_look_back, data.shape[1], data.shape[2], data.shape[3]), dtype='float32')
        
        for i in range(0, n_samples):
            tmp = data[i:i+n_look_back]
            if tmp.shape[0] == n_look_back:
                for rotate_channel_id in range(0, n_look_back): # rotate the channels such that bring the current input as first channel
                    tmp[rotate_channel_id] = tmp[n_look_back-1-rotate_channel_id]
                tmp = tmp.reshape(1, n_look_back, data.shape[1], data.shape[2], data.shape[3])
                data_5d[k] = tmp
                tmp = [] # clear tmp
                k = k + 1
    
        return data_5d
    
    
    def generate(self, train_dir, dataset_dir, trainStart, trainEnd):
        framesBack= 5
        void_label = -1.
        
        # Given ground-truths, load training frames
        # ground-truths end with '*.png'
        # training frames end with '*.jpg'
        
        # given ground-truths, load inputs  
        Y_list = glob.glob(os.path.join(train_dir, '*.png'))
        X_list= glob.glob(os.path.join(dataset_dir,'*.jpg'))
    
    
        if len(Y_list)<=0 or len(X_list)<=0:
            raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')
          
        if len(X_list)!=len(Y_list):
            raise ValueError('The number of X_list and Y_list must be equal.')   
            
        # X must be corresponded to Y
        X_list = sorted(X_list)
        Y_list = sorted(Y_list)
        
        #Solo quedarme con ls imagener comprendidas en el intervalo trainStart - trainEnd
        X_list = X_list[trainStart:trainEnd]
        Y_list = Y_list[trainStart:trainEnd]
        #Y_list = Y_list[trainStart+framesBack-1:trainEnd]
        
        # load training data
        X = []
        Y = []
        #np.set_printoptions(threshold=sys.maxsize)
    
        for i in range(len(X_list)):
            x = kImage.load_img(X_list[i], target_size=(self.maxW, self.maxH))
            x = kImage.img_to_array(x)
            X.append(x)
         
        #for i in range(framesBack-1,len(Y_list)):
        for i in range(len(Y_list)):
            x = kImage.load_img(Y_list[i], grayscale = True, target_size=(self.maxW, self.maxH))
            x = kImage.img_to_array(x)
            shape = x.shape
            x/=255.0
            x = x.reshape(-1)
            idx = np.where(np.logical_and(x>0.25, x<0.8))[0] # find non-ROI
            if (len(idx)>0):
                x[idx] = void_label
            x = x.reshape(shape)
            x = np.floor(x)
            Y.append(x)
            
        X = np.asarray(X)
        Y = np.asarray(Y)
           
        
        #X = generateTemporalInformation(X, framesBack)
        X = self.make_5d(X, framesBack)
        print(str(X.shape))
        
        #Descomentar para usar El lstm con 5doutput(sin Dense)
        Y = self.make_5d(Y, framesBack)
        print(str(Y.shape))
        
        # compute class weights
        cls_weight_list = []
        for i in range(Y.shape[0]):
            y = Y[i].reshape(-1)
            idx = np.where(y!=void_label)[0]
            if(len(idx)>0):
                y = y[idx]
            lb = np.unique(y) #  0., 1
            cls_weight = compute_class_weight('balanced', lb , y)
            class_0 = cls_weight[0]
            class_1 = cls_weight[1] if len(lb)>1 else 1.0
            
            cls_weight_dict = {0:class_0, 1: class_1}
            cls_weight_list.append(cls_weight_dict)
    
        cls_weight_list = np.asarray(cls_weight_list)
        
        return [X, Y, cls_weight_list]