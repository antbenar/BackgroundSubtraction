from config import Init
from train import ModelTrain

class Main():
    """ Constructor """
    def __init__(self,init):
        super().__init__()
        self.model = ModelTrain(init)

    def train(self):
        self.model.execute()
    

if __name__ == "__main__":
    # Setting  
    init    = Init()

    main = Main(init)
    main.train()
