import torch
import logging
from config                                  import Settings
from trainTest                               import ModelTrainTest
from Model.Base_model                        import Net
from Model.Base2D.Base2D_model               import Net2D
from Model.Multiscale2D.Multiscale2D_model   import MultiscaleNet2D
from Model.LoGo2D.LoGo2D_model               import LoGoNet2D

class Main():
    """ Constructor """
    def __init__(self,settings):
        super().__init__()
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        self.settings = settings
        self._init_model()
        
        
    #----------------------------------------------------------------------------------------
    # Function to train the model
    # mode='train_val', 'test', 'train_val_test'
    #----------------------------------------------------------------------------------------
        
    def execute(self, mode='train_val'):
        if(mode=='test'):
            self.settings.dataset_fg_bg = False
            
        self.model = ModelTrainTest(self.net, self.settings)
        self.model.execute(mode) 
        
        
    #----------------------------------------------------------------------------------------
    # Function to save to tensorboard the model and a sample of the dataset
    #----------------------------------------------------------------------------------------
    
    def saveTrainData(self):
        self.model = ModelTrainTest(self.net, self.settings)
        self.model.saveTrainData()
        
        del self.model
        torch.cuda.empty_cache()
        
            
    #----------------------------------------------------------------------------------------
    # Function to calculate the size of the model
    #----------------------------------------------------------------------------------------
    
    def calculateModelSize(self):
        self.settings.device = torch.device('cpu')
        self.model  = ModelTrainTest(self.net, self.settings)
        self.model.calculateModelSize()
        
        del self.model
        torch.cuda.empty_cache()
        
        
    #----------------------------------------------------------------------------------------
    # Instance Model
    #----------------------------------------------------------------------------------------
    
    def _init_model(self):
        if (self.settings.model_name == 'Model_M1-2D-LSTM'):
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = False
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'sigmoid'
            self.settings.up_mode             = 'base'
            self.net = Net2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = self.settings.up_mode,
                activation = self.settings.activation
            )
        elif (self.settings.model_name == 'Model_M1-2D-LSTM_priority'):
            self.settings.priority_active              = True
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = False
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'sigmoid'
            self.settings.up_mode             = 'base'
            self.net = Net2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = self.settings.up_mode,
                activation = self.settings.activation
            )
        elif (self.settings.model_name == 'Model_M1-2D-LSTM_softmax' or self.settings.model_name == 'Model_M1-2D-LSTM_softmax_dobleLoss'):
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = True
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'softmax'
            self.settings.up_mode             = 'base'
            self.net = Net2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = self.settings.up_mode,
                activation = self.settings.activation
            )
        elif (self.settings.model_name == 'Model_Multiscale-2D'):
            # parameters of the dataset
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = False
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'sigmoid'
            self.settings.up_mode             = 'base'
            # model
            self.net = MultiscaleNet2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = 'base',
                activation = 'sigmoid'
            )
        elif (self.settings.model_name == 'Model_Multiscale-2D_softmax'):
            # parameters of the dataset
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = True
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'softmax'
            self.settings.up_mode             = 'base'
            # model
            self.net = MultiscaleNet2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = 'base',
                activation = 'softmax'
            )
        elif (self.settings.model_name == 'Model_Multiscale_M2-2D_softmax'):
            # parameters of the dataset
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = True
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'softmax'
            self.settings.up_mode             = 'M2'
            # model
            self.net = MultiscaleNet2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = self.settings.up_mode,
                activation = 'softmax'
            )
        elif (self.settings.model_name == 'Model_Multiscale-2D_softmax_noattention'):
            # parameters of the dataset
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = True
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'softmax'
            self.settings.up_mode             = 'base'
            # model
            self.net = MultiscaleNet2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = 'base',
                activation = 'softmax',
                attention  = False
            )
        elif (self.settings.model_name == 'Model_M2'):
            self.settings.framesBack          = self.settings.framesBack
            self.settings.dataset_fg_bg       = False
            self.settings.model_dim           = '3D'
            self.settings.up_mode             = 'M2'
            self.settings.activation          = 'sigmoid'
            self.net = Net(
                self.settings.n_channels, 
                self.settings.p_dropout, 
                up_mode= self.settings.up_mode
            )
        elif (self.settings.model_name == 'Model_M1'):
            self.settings.framesBack          = self.settings.framesBack
            self.settings.dataset_fg_bg       = False
            self.settings.model_dim           = '3D'
            self.settings.up_mode             = 'base'
            self.settings.activation          = 'sigmoid'
            self.net = Net(
                self.settings.n_channels, 
                self.settings.p_dropout, 
                up_mode= self.settings.up_mode
            )  
        elif (self.settings.model_name == 'Model_LoGo2D'):
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = False
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'sigmoid'
            self.settings.up_mode             = 'base' 
            self.net = LoGoNet2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = self.settings.up_mode,
                activation = self.settings.activation
            )
        else:
            raise NotImplementedError('Model is not implemented')
        
        
        
if __name__ == "__main__":
    
    # models = [ 'Model_M1-2D-LSTM', 'Model_M1-2D-LSTM_softmax',
    #           'Model_Multiscale-2D', 'Model_Multiscale-2D_softmax', 'Model_Multiscale_M2-2D_softmax', 
    #           'Model_Multiscale-2D_softmax_noattention', 'Model_LoGo2D', 'Model_M1-2D-LSTM_softmax_dobleLoss']
    
    # models = ['Model_M1-2D-LSTM', 'Model_M1-2D-LSTM_softmax','Model_Multiscale-2D', 'Model_Multiscale-2D_softmax']
     
    models = ['Model_M1-2D-LSTM_priority']
    
    for model in models:
        settings    = Settings(model)
        main        = Main(settings)
        # main.execute(mode='train_val')
        main.execute(mode='test')
        # main.saveTrainData()
        #main.calculateModelSize()
    