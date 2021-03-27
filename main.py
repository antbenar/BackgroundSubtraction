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
    #----------------------------------------------------------------------------------------
        
    def train(self):
        self.model = ModelTrainTest(self.net, self.settings)
        self.model.execute(mode='train_val_test')
        
    #----------------------------------------------------------------------------------------
    # Function to test the model
    #----------------------------------------------------------------------------------------
        
    def test(self):
        self.model = ModelTrainTest(self.net, self.settings)
        self.model.load(self.settings.loadPath)
        self.model.execute(mode='test')
        
    #----------------------------------------------------------------------------------------
    # Function to train the model
    #----------------------------------------------------------------------------------------
        
    def run(self):
        self.model = ModelTrainTest(self.net, self.settings)
        self.model.load(self.settings.loadPath)
        #self.model.execute()
        
        
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
        if (self.settings.model_dim == '2D'):
            self.settings.framesBack = 0
            self.net = Net2D(
                self.settings.n_channels, 
                self.settings.p_dropout
            )
        elif (self.settings.model_dim == '3D'): #3D
            self.net = Net(
                self.settings.n_channels, 
                self.settings.p_dropout, 
                up_mode= self.settings.up_mode
            )
        else:
            raise NotImplementedError('Dimension model not implemented')
        
        
        
if __name__ == "__main__":
    # Setting  
    settings    = Settings()

    main = Main(settings)
    
    # Instance model
    main._init_model()
    
    #main.test()
    main.train()
    #main.saveTrainData()
    #main.calculateModelSize()
    
