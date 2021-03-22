import logging
from config import Init
from train  import ModelTrain

class Main():
    """ Constructor """
    def __init__(self,init):
        super().__init__()
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
    #----------------------------------------------------------------------------------------
    # Function to train the model
    #----------------------------------------------------------------------------------------
        
    def train(self):
        self.model = ModelTrain(init)
        self.model.execute()
    
    def saveTrainData(self):
        self.model = ModelTrain(init)
        self.model.saveTrainData()
        
        
if __name__ == "__main__":
    # Setting  
    init    = Init()

    main = Main(init)
    main.train()
    #main.saveTrainData()
