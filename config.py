import torch
import os
from datetime import datetime

class Init(object):
    
    def __init__(self):
        super().__init__()
        
        # Device
        #self.device_name        = 'cpu'
        self.device_name        = 'cuda:0'
        #self.device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device             = torch.device(self.device_name)
        
        self.num_workers        =  0
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~ settings model ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.model_name         = 'AtentionUnetLSTM_model'
        self.up_mode            = 'base'
        self.n_channels         = 3
        self.p_dropout          = 0.2
        self.threshold          = 0.75
        
        # train
        self.epochs             = 50
        self.batch_size         = 5
        self.shuffle            = False
        self.train_result_dir   = os.path.join('TrainResult','Model_M3_2D')

        # optimizer
        self.beta_a             = 0.9
        self.beta_b             = 0.999
        self.lr                 = 1e-4
        
        # tensorboard
        self.logdir             = "logs_tensorboard/New_results/Training_lossM1_2D_766fm_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "/"
        self.view_batch         = 100                                     # save losses every n batches
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~ settings dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.dataset_dir        = os.path.join('..', '..', 'CDnet2014_dataset')
        self.data_format        = 'channels_last'
        self.framesBack         = 5
        self.resize             = True  #If true, uses trainStart - trainEnd to dataset
        self.trainStart         = 1
        self.trainEnd           = 100
        
        # splits of dataset
        self.train_split        = 0.6
        self.val_split          = 0.1
        
        # dictionary of catergories and scenes of the Cdnet2014
        
        self.dataset            = {
            'baseline':['highway']
        }
        
        """self.dataset         = {
            'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
            'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
            'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
            'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
            'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
            'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
            'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
            'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
        }"""