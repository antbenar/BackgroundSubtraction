import torch
import logging
from config import Init
from train  import ModelTrain

class Main():
    """ Constructor """
    def __init__(self,init):
        super().__init__()
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        self.init = init
        
    #----------------------------------------------------------------------------------------
    # Function to train the model
    #----------------------------------------------------------------------------------------
        
    def train(self):
        self.model = ModelTrain(self.init)
        self.model.execute()

        
    #----------------------------------------------------------------------------------------
    # Function to save to tensorboard the model and a sample of the dataset
    #----------------------------------------------------------------------------------------
    
    def saveTrainData(self):
        self.model = ModelTrain(self.init)
        self.model.saveTrainData()
        
            
    #----------------------------------------------------------------------------------------
    # Function to calculate the size of the model
    #----------------------------------------------------------------------------------------
    
    def calculateModelSize(self):
        self.init.device = torch.device('cpu')
        self.model  = ModelTrain(self.init)
        self.model.calculateModelSize()
        
        
if __name__ == "__main__":
    # Setting  
    init    = Init()

    main = Main(init)
    #main.train()
    #main.saveTrainData()
    main.calculateModelSize()