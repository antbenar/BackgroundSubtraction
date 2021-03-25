import torch
import logging
from config import Settings
from train  import ModelTrain

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
        self.model = ModelTrain(self.settings)
        self.model.execute()

        
    #----------------------------------------------------------------------------------------
    # Function to save to tensorboard the model and a sample of the dataset
    #----------------------------------------------------------------------------------------
    
    def saveTrainData(self):
        self.model = ModelTrain(self.settings)
        self.model.saveTrainData()
        
            
    #----------------------------------------------------------------------------------------
    # Function to calculate the size of the model
    #----------------------------------------------------------------------------------------
    
    def calculateModelSize(self):
        self.init.device = torch.device('cpu')
        self.model  = ModelTrain(self.settings)
        self.model.calculateModelSize()
        
        
if __name__ == "__main__":
    # Setting  
    settings    = Settings()

    main = Main(settings)
    main.train()
    #main.saveTrainData()
    #main.calculateModelSize()