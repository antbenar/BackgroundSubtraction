import torch
import os

class Init(object):
    
    def __init__(self):
        super().__init__()
        
        # Device
        self.device_name        = 'cuda:0'
        self.device             = torch.device(self.device_name)
        
        self.num_workers        =  0
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~ settings model ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.model_name         = 'AtentionUnetLSTM_model'
        
        self.n_channels         = 3
        self.p_dropout          = 0.2
        #train
        self.epochs             = 50
        self.val_split          = 0.2
        self.batch_size         = 2
        self.shuffle            = False
        self.lr                 = 1e-4
        self.train_result_dir   = os.path.join('TrainResult')
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~ settings dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.dataset_dir        = os.path.join('..', '..', 'CDnet2014_dataset')
        self.data_format        = 'channels_last'
        self.framesBack         = 5
        self.trainStart         = 1
        self.trainEnd           = 20
        
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