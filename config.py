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
        
        self.model_name         = 'Model_M1'
        self.model_dim          = '3D'   # dimension of the model
        self.up_mode            = 'base' # only used if model_dim = 3D
        self.n_channels         = 3      # input chanels
        self.p_dropout          = 0.2    # dropout probability
        self.threshold          = 0.75   # Static threshold to make segmentation
        
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
        self.loadPath           = os.path.join('TrainResult', self.model_name , 'baseline', 'mdl_baseline_highway49.pth')
        self.plot_test          = True
        
        # tensorboard
        self.logdir             = "logs_tensorboard/New_results/Training_lossM1_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "/"
        self.view_batch         = 100                                     # print losses every n batches
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~ settings dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.dataset_dir        = os.path.join('..', '..', 'CDnet2014_dataset')
        self.data_format        = 'channels_last'
        self.framesBack         = 5
        
        self.resize             = True   # If the frames will be resized
        self.width              = 240    # width of the frame
        self.height             = 320    # height of the frame
        
        self.dataset_range      = False   # If true, uses trainStart - trainEnd to dataset
        self.trainStart         = 1
        self.trainEnd           = 100
        
        # splits of dataset
        self.train_split        = 0.6
        self.val_split          = 0.1
        
        # dictionary of catergories and scenes of the Cdnet2014
        
        # self.dataset            = {
        #     'baseline':['highway']
        # }
        
        self.dataset            = {
            'baseline':['PETS2006'],
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass']
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