import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from keras.preprocessing import image as kImage

class GenerateData(Dataset):
    
    def __init__(
                     self, 
                     dataset_gt_dir, 
                     dataset_dir, 
                     framesBack, 
                     trainStart, 
                     trainEnd, 
                     transform=None, 
                     data_format='channels_last',
                     void_value = True,
                     showSample = False,
                     resize = False
                 ):
        """
        Args:
            dataset_gt_dir (string): Path to the input dir.
            dataset_dir (string): Path to the groundtruth dir.
            framesBack (int): Number of frames_back in out temporal subsets
            trainStart (int): Index of the first element to take into our data set
            trainEnd (int): Index of the last element to take into our data set
            transform: Null
            data_format (string): If the chanels are in last dim
            showSample (Boolean): If true plot the intermediate frame to qualitatively validate our trainset
        """
        self.dataset_gt_dir = dataset_gt_dir
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.data_format = data_format
        
        self.framesBack = framesBack
        self.resize     = resize
        self.trainStart = trainStart
        self.trainEnd   = trainEnd
        self.void_label = -1.
        self.void_value = void_value
        self.dataset    = self.generate()
        
        if(showSample):
            ## plot the intermediate frame to qualitatively validate our trainset
            self.plotSample(idx_frame=(self.trainEnd-self.trainStart)//2)
    
    
    #----------------------------------------------------------------------------------------
    # Function to get length of the dataset
    #----------------------------------------------------------------------------------------
    
    def __len__(self):
        return len(self.dataset[0])
    
    
    #----------------------------------------------------------------------------------------
    # function to retrieve an element by index
    #----------------------------------------------------------------------------------------
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.dataset[0][idx]
        gt = self.dataset[1][idx]
        
        sample = {'inputs': input, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    #----------------------------------------------------------------------------------------
    # Given ground-truths, load training frames
    # ground-truths end with '*.png'
    # training frames end with '*.jpg'
    
    # given ground-truths, load inputs  
    #----------------------------------------------------------------------------------------
    
    def generate(self):

        #load imgs dir
        Y_list = glob.glob(os.path.join(self.dataset_gt_dir, '*.png'))
        X_list= glob.glob(os.path.join(self.dataset_dir,'*.jpg'))
    
        if len(Y_list)<=0 or len(X_list)<=0:
            raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')
          
        if len(X_list)!=len(Y_list):
            raise ValueError('The number of X_list and Y_list must be equal.')   
            
        # X must be corresponded to Y
        X_list = sorted(X_list)
        Y_list = sorted(Y_list)
        
        if(self.resize == True):
            #Solo quedarme con las imagenes comprendidas en el intervalo trainStart - trainEnd
            X_list = X_list[self.trainStart:self.trainEnd]
            Y_list = Y_list[self.trainStart:self.trainEnd]
        
        # load training data
        self.X = self.loadImages(X_list)
        self.Y = self.loadImagesGroundtruth(Y_list)
        
        
        #Given a sequence of frames, divide in groups of five consecutive frames
        self.X = self.make_5d(self.X)
        self.Y = self.make_5d(self.Y)
        
        if (self.data_format=='channels_last'):    
            #  move chanels to the front of the tensor
            self.X = np.moveaxis(self.X, -1, 1)
            self.Y = np.moveaxis(self.Y, -1, 1)
        
        return [self.X, self.Y]
    
    
    #----------------------------------------------------------------------------------------
    # Given a sequence of frames, divide in groups of five consecutive frames
    #----------------------------------------------------------------------------------------
    
    def make_5d(self, data):
        n_look_back = self.framesBack
        
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
    
    #----------------------------------------------------------------------------------------
    # Load images from directory to an array
    #----------------------------------------------------------------------------------------
    
    def loadImages(self, X_list):
        X = []
    
        for i in range(len(X_list)):
            #img = io.imread(X_list[i])
            #img = kImage.load_img(X_list[i], target_size=(self.maxW, self.maxH))
            img = kImage.load_img(X_list[i])
            img = kImage.img_to_array(img)
            X.append(img)
            
        return np.asarray(X)
    
    #----------------------------------------------------------------------------------------
    # Load images from directory to an array (void pixels are identified with -1)
    #----------------------------------------------------------------------------------------
    
    def loadImagesGroundtruth(self, Y_list):
        Y = []
    
        for i in range(len(Y_list)):
            img = kImage.load_img(Y_list[i], color_mode = "grayscale")
            img = kImage.img_to_array(img)
            
            if(self.void_value):
                shape = img.shape
                img/=255.0
                img = img.reshape(-1)
                idx = np.where(np.logical_and(img>0.25, img<0.8))[0] # find non-ROI
                if (len(idx)>0):
                    img[idx] = self.void_label
                img = img.reshape(shape)
                img = np.floor(img)
            
            Y.append(img)
            
        return np.asarray(Y)
    
    #----------------------------------------------------------------------------------------
    # Plot some samples of the dataset
    #----------------------------------------------------------------------------------------
    
    def plotSample(self, idx_frame):     
        print("~~~~~~~~~~~~~~~ Plot Sample ~~~~~~~~~~~~~~~~")
        print('Input shape', self.dataset[0].shape)
        print('Gt shape', self.dataset[1].shape)
        
        inputs, gt = self.dataset[0][idx_frame], self.dataset[1][idx_frame]
        
        if (self.data_format=='channels_last'):
            inputs = np.moveaxis(inputs, 0, -1)
            gt = np.moveaxis(gt, 0, -1)
           
        # change -1 to a gray color
        if(self.void_value):
            shape = gt.shape
            gt/=255.0
            gt = gt.reshape(-1)
            idx = np.where(gt==-1)[0] # find non-ROI
            if (len(idx)>0):
                gt[idx] = 0.55
            gt = gt.reshape(shape)
    
        ax = plt.subplot(1, 2, 1)
        ax.set_title('Input #{}'.format(idx_frame))
        ax.axis('off')
        plt.imshow(inputs[0]/255)
        plt.tight_layout()
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Gt #{}'.format(idx_frame))
        ax2.axis('off')
        plt.imshow(gt[0], cmap=plt.get_cmap('gray'))
        plt.tight_layout()

        plt.show()
        
        
