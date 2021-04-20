import torch
import os
from datetime import datetime

class Settings(object):
    
    def __init__(self):
        super().__init__()
        
        # Device
        #self.device_name        = 'cpu'
        self.device_name        = 'cuda:0'
        #self.device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device             = torch.device(self.device_name)
        
        self.num_workers        =  0
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~ settings model ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.model_name         = 'Model_Multiscale-2D_softmax_noattention'
        self.model_dim          = '2D'      # dimension of the model
        self.up_mode            = 'base'    # It changes in the main according to model_name ('base', 'M2', 'M3')
        self.activation         = 'sigmoid' # only used if model_dim = 2D ('softmax' or 'sigmoid')
        
        self.n_channels         = 3         # input chanels
        self.p_dropout          = 0.2       # dropout probability
        self.threshold          = 0.75      # Static threshold to make segmentation
        
        # train
        self.epochs             = 50
        self.batch_size         = 5
        self.shuffle            = False
        self.train_result_dir   = os.path.join('TrainResult',self.model_name)

        # optimizer
        self.beta_a             = 0.9
        self.beta_b             = 0.999
        self.lr                 = 1e-4
        
        #scheduler
        self.scheduler_active   = False  # If true uses lr_decay_steps and lr_decay_factor
        self.lr_decay_steps     = 1e-4
        self.lr_decay_factor    = 1e-4
        
        # loader - test
        #self.loadPath           = os.path.join('TrainResult', self.model_name , 'baseline', 'mdl_baseline_highway49.pth')
        self.loadPath           = os.path.join('TrainResult', self.model_name)
        self.plot_test          = False
        self.log_test           = True
        
        
        # tensorboard
        #self.logdir             = "logs_tensorboard/New_results/" + self.model_name+"_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "/"
        self.logdir             = "logs_tensorboard/__results_all_dirs/" + self.model_name +"/"
        self.view_batch         = 10                                     # print losses every n batches
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~ settings dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.dataset_dir        = os.path.join('..', '..', 'CDnet2014_dataset')
        self.data_format        = 'channels_last'
        self.framesBack         = 5
        self.differenceFrames   = True    # Used only when framesBack > 0, It additions the framesback with the difference of the current frame
        self.showSample         = True
        self.dataset_fg_bg      = False   # To generate a 2 chanel gt, one for background an another one for foreground
        
        self.resize             = True    # If the frames will be resized
        self.width              = 240     # width of the frame
        self.height             = 320     # height of the frame
        
        self.dataset_range      = False    # If true, uses trainStart - trainEnd to dataset
        self.trainStart         = 200 
        self.trainEnd           = 260
        
        # splits of dataset
        self.train_split        = 0.6
        self.val_split          = 0.1
        
        
        # dictionary of catergories and scenes of the Cdnet2014
        
        #  -- REVISAR BOATS (MUY LARGA)
        #  -- REVISAR FOUNTAIN, CARGA MAL EL GROUNDTRUTH
        
        # self.dataset            = {
        #     'dynamicBackground':['fall']
        # }
        
        # self.dataset            = {
        #     'baseline':['highway']
        # }
        
        
        self.dataset            = {
            'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
            'dynamicBackground':['canoe', 'fall']
        }
        
        # self.dataset            = {
        #     'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
        #     'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
        #     'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
        #     'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
        #     'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
        #     'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
        #     'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
        #     'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
        #     'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
        #     'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
        #     'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
        # }