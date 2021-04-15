import torch
import logging
from config                      import Settings
from trainTest                   import ModelTrainTest
from Model.Base_model            import Net
from Model.Base2D.Base2D_model   import Net2D
#from Model.Unet2D.Unet2D_model  import Net
from termcolor                   import colored

class Main():
    """ Constructor """
    def __init__(self,settings):
        super().__init__()
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        self.settings = settings
        
        
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
        elif (self.settings.model_name == 'Model_M1-2D-LSTM_softmax'):
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
        elif (self.settings.model_name == 'Model_M2-2D-LSTM'):
            #not yet implemented
            self.settings.framesBack          = 0
            self.settings.dataset_fg_bg       = False
            self.settings.model_dim           = '2D'
            self.settings.activation          = 'sigmoid'
            self.settings.up_mode             = 'M2'
            self.net = Net2D(
                self.settings.n_channels, 
                self.settings.p_dropout,
                up_mode    = self.settings.up_mode,
                activation = self.settings.activation
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
        else:
            raise NotImplementedError('Model is not implemented')
        
        
        
if __name__ == "__main__":
    # Setting  
    settings    = Settings()

    main = Main(settings)
    
    # Instance model
    main._init_model()
    
    #main.execute(mode='train_val')
    main.execute(mode='test')
    #main.execute(mode='train_val_test')
    #main.saveTrainData()
    #main.calculateModelSize()
    
