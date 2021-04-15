import torch
import numpy as np
import os
import glob
from torch.utils.data          import Dataset
from matplotlib                import pyplot as plt
from keras.preprocessing       import image as kImage
from torch.utils.data.sampler  import SubsetRandomSampler

class GenerateData(Dataset):
    
    def __init__(
                     self, 
                     dataset_gt_dir, 
                     dataset_dir, 
                     framesBack, 
                     resize,
                     width,
                     height,
                     dataset_range,
                     trainStart, 
                     trainEnd, 
                     transform        = None, 
                     data_format      = 'channels_last',
                     dataset_fg_bg    = False,
                     void_value       = True,
                     showSample       = False,
                     differenceFrames = False
                 ):
        """
        Args:
            dataset_gt_dir (string): Path to the input dir.
            dataset_dir (string)   : Path to the groundtruth dir.
            framesBack (int)       : Number of frames_back in out temporal subsets, if It is 0, the datase only will be a4D tensor (BHWC)
            resize (int)           : If the frames will be resized
            width (int)            : width of the frame
            height (int)           : height of the frame
            dataset_range (bool)   : If the dataset has a range
            trainStart (int)       : Index of the first element to take into our data set
            trainEnd (int)         : Index of the last element to take into our data set
            transform              : Null
            data_format (string)   : If the chanels are in last dim
            dataset_fg_bg (Boolean): To generate a 2 chanel gt, one for background an another one for foreground
            showSample (Boolean)   : If true plot the intermediate frame to qualitatively validate our trainset
            differenceFrames (Boolean): If true It additions the framesback with the difference of the current frame
        """
        
        self.dataset_gt_dir = dataset_gt_dir
        self.dataset_dir    = dataset_dir
        self.transform      = transform
        self.data_format    = data_format
        self.dataset_fg_bg  = dataset_fg_bg
        self.differenceFrames  = differenceFrames
        
        self.resize         = resize
        self.width          = width
        self.height         = height
        
        self.dataset_range  = dataset_range
        self.trainStart     = trainStart
        self.trainEnd       = trainEnd
        
        self.framesBack     = framesBack
        self.void_label     = -1.
        self.void_value     = void_value
        
        # generate dataset
        self.dataset        = self.generate()
        
        print('Dataset Input shape', self.dataset[0].shape)
        print('Dataset Gt shape', self.dataset[1].shape)
        
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
        
        if(self.dataset_range == True):
            #Solo quedarme con las imagenes comprendidas en el intervalo trainStart - trainEnd
            X_list = X_list[self.trainStart:self.trainEnd]
            Y_list = Y_list[self.trainStart:self.trainEnd]
        
        # load training data
        self.X = self.loadImages(X_list)
        self.Y = self.loadImagesGroundtruth(Y_list)
        
        
        #Given a sequence of frames, divide in groups of five consecutive frames
        if (self.framesBack > 0):    
            self.X = self.make_5d(self.X)
            self.Y = self.make_5d(self.Y)
        
        
        if (self.data_format== 'channels_last'):    
            #  move chanels to the front of the tensor (B, T, H, W, C) -> (B, C, T, H, W)
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
            tmp  = data[i:i+n_look_back]
            tmp2 = data[i:i+n_look_back]
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
            if(self.resize):
                img = kImage.load_img(X_list[i], target_size=(self.width, self.height))
            else:
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
            if(self.resize):
                img = kImage.load_img(Y_list[i], target_size=(self.width, self.height), color_mode = "grayscale")
            else:
                img = kImage.load_img(Y_list[i], color_mode = "grayscale")
            
            img = kImage.img_to_array(img)
            
            if(self.void_value):
                shape = img.shape
                img/=255.0
                img = img.reshape(-1)
                idx = np.where(np.logical_and(img>0, img<1))[0] # find non-ROI
                if (len(idx)>0):
                    img[idx] = self.void_label
                img = img.reshape(shape)

            
            if (self.dataset_fg_bg):
                img_fg          = np.copy(img)
                img_bg          = img
                shape           = img_bg.shape
                img_bg          = img_bg.reshape(-1)
                idx_bg          = np.where(img_bg==0)[0] 
                idx_fg          = np.where(img_bg==1)[0] 
                
                img_bg[idx_bg]  = 1
                img_bg[idx_fg]  = 0
                img_bg          = img_bg.reshape(shape)
                 
                img = np.concatenate([img_bg, img_fg], axis=2)
            
            Y.append(img)
            
        return np.asarray(Y)
    
    
    #----------------------------------------------------------------------------------------
    # Function to separate the whole dataset into three subsets for training, validation of training and test
    #----------------------------------------------------------------------------------------
        
    def train_val_test_split(self, train_split_, val_split_, shuffle, batch_size, num_workers): 

        dataset_size = len(self.dataset[0])
        indices      = list(range(dataset_size))
        train_split  = int(np.floor(train_split_ * dataset_size))                # obtain the number of train samples
        val_split    = int(np.floor(train_split_ * val_split_ * dataset_size))   # obtain the number of val samples
        train_split  = train_split - val_split                                   # obtain the position where the train samples end
        val_split    = train_split + 2*val_split                                 # obtain the position where the val samples end
        
        if shuffle :
            np.random.seed(1234)
            indices = np.random.shuffle(indices)
            
        train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]
        
        
        train_loader = torch.utils.data.DataLoader(
                            self, 
                            batch_size  = batch_size, 
                            num_workers = num_workers,
                            sampler     = train_indices
                        )
        
        val_loader = torch.utils.data.DataLoader(
                            self, 
                            batch_size  = batch_size, 
                            num_workers = num_workers,
                            sampler     = val_indices
                        )
        
        test_loader = torch.utils.data.DataLoader(
                            self, 
                            batch_size  = batch_size, 
                            num_workers = num_workers,
                            sampler     = test_indices
                        )
        
        return train_loader, val_loader, test_loader
    
    
    #----------------------------------------------------------------------------------------
    # Plot some samples of the dataset
    #----------------------------------------------------------------------------------------
    
    def plotSample(self, idx_frame):             
        inputs, gt = self.dataset[0][idx_frame], self.dataset[1][idx_frame]

        if (self.data_format=='channels_last'):
            inputs = np.moveaxis(inputs, 0, -1)
            gt = np.moveaxis(gt, 0, -1)
           
        # change -1 to a gray color
        if(self.void_value):
            shape = gt.shape
            gt    = gt.reshape(-1)
            idx   = np.where(gt==-1)[0] # find non-ROI
            if (len(idx)>0):
                gt[idx] = 0.55
            gt = gt.reshape(shape)
    
        # if there is an stack of frames, get the first frame
        if (self.framesBack > 0):   
            inputs = inputs[0]
            gt     = gt[0]
            

        
        
        
        if(self.dataset_fg_bg):
            gt        = np.moveaxis(gt, -1, 0)
            
            shape     = (1, self.width, self.height)
            img_bg    = gt[0].reshape(shape)
            img_fg    = gt[1].reshape(shape)

            img_bg    = np.moveaxis(img_bg, 0, -1)
            img_fg    = np.moveaxis(img_fg, 0, -1)
            
            ax = plt.subplot(1, 3, 1)
            ax.set_title('Input #{}'.format(idx_frame))
            ax.axis('off')
            plt.imshow(inputs/255)
            plt.tight_layout()

            ax2 = plt.subplot(1, 3, 2)
            ax2.set_title('Gt Bg #{}'.format(idx_frame))
            ax2.axis('off')
            plt.imshow(img_bg, cmap=plt.get_cmap('gray'))
            plt.tight_layout()
            
            ax3 = plt.subplot(1, 3, 3)
            ax3.set_title('Gt Fg #{}'.format(idx_frame))
            ax3.axis('off')
            plt.imshow(img_fg, cmap=plt.get_cmap('gray'))
            plt.tight_layout()
    
            plt.show()
        else:
            ax = plt.subplot(1, 2, 1)
            ax.set_title('Input #{}'.format(idx_frame))
            ax.axis('off')
            plt.imshow(inputs/255)
            plt.tight_layout()
        
            ax2 = plt.subplot(1, 2, 2)
            ax2.set_title('Gt #{}'.format(idx_frame))
            ax2.axis('off')
            plt.imshow(gt, cmap=plt.get_cmap('gray'))
            plt.tight_layout()
    
            plt.show()
