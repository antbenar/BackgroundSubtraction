import os
import torch
import torch.nn   as nn
from Model.model                         import Net
from Dataset.generateData                import GenerateData
from Tensorboard.TensorboardTool         import TensorBoardTool

class ModelTrain(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.init = init
        
        # model static attributes
        self.n_channels = init.n_channels
        self.p_dropout = init.p_dropout
        self.device = init.device
                
        # instance the model
        self.net = Net(
            self.n_channels, 
            self.p_dropout
        )
        # net to device
        self.net.to(self.device)
        
        
    #----------------------------------------------------------------------------------------
    # Function to train the model
    #----------------------------------------------------------------------------------------
        
    def train(self, trainloader, train_result_dir):
        criterion_loss = nn.MSELoss(reduction='sum')
        #criterion_loss = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.init.lr, betas=(self.init.beta_a, self.init.beta_b))
        
        # create tensorboard tool
        tensorBoardTool = TensorBoardTool(self.init.logdir)        
                
        for epoch in range(self.init.epochs):  # loop over the dataset multiple times
            print("Epoch = ", epoch+1)
            running_loss = 0.0
            
            for i_batch, sample_batched  in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs      = sample_batched['inputs'].to(self.device)
                groundtruth = sample_batched['gt'].to(self.device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion_loss(outputs, groundtruth)
                loss.backward()
                optimizer.step()
        
                # print statistics every num_stat_batches
                running_loss += loss.item()
                num_stat_batches = 2
                if (i_batch+1) % num_stat_batches == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i_batch + 1, running_loss / num_stat_batches))
                    
                    tensorBoardTool.saveTrainLoss(
                        name_folder = self.init.train_loss_dir,
                        tag         = self.init.train_loss_tag,
                        loss        = running_loss / num_stat_batches, 
                        step        = epoch * len(trainloader) + i_batch
                    )
                    
                    running_loss = 0.0
                    
                del loss, outputs
                
        
        print('Finished Training')
        
        # run tensorboard
        tensorBoardTool.closeWriter()
        tensorBoardTool.run()
        
        # save the model
        torch.save(self.net.state_dict(), train_result_dir)
        
    #----------------------------------------------------------------------------------------
    # Function to iterate over categories and scenes and train the model for each one
    #----------------------------------------------------------------------------------------
        
    def execute(self):
        
        print('~~~~~~~~~~~~~~ Current method >>> ' + self.init.model_name)
        
        if not os.path.exists(self.init.train_result_dir):
            os.makedirs(self.init.train_result_dir)
                
        # Go through each scene
        for category, scene_list in self.init.dataset.items():     
            for scene in scene_list: 
                
                #~~~~~~~~~~~~~~~~~~~~~ Load dataset for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                print("~~~~~~~ Generating data ->>> " + category + " / " + scene + " ~~~~~~~~~~")

                dataset_dir = os.path.join(self.init.dataset_dir, category, scene, 'input')
                dataset_gt_dir = os.path.join(self.init.dataset_dir, category, scene, 'groundtruth')
                
                trainset = GenerateData(
                    dataset_gt_dir, dataset_dir,
                    framesBack=self.init.framesBack,
                    trainStart=self.init.trainStart,
                    trainEnd=self.init.trainEnd,
                    data_format=self.init.data_format,
                    showSample = True
                )
                
                trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=self.init.batch_size, 
                    shuffle=self.init.shuffle, 
                    num_workers=self.init.num_workers
                )
                
                #~~~~~~~~~~~~~~~~~~~~~ Train net for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                print("~~~~~~~ Training ->>> " + category + " / " + scene + " ~~~~~~~~~~")
                train_result_dir = os.path.join(self.init.train_result_dir, 'mdl_' + category + '_' + scene + '.h5')
                self.train(trainloader, train_result_dir)
                del trainset
          
    #----------------------------------------------------------------------------------------
    # Function to iterate over categories and scenes to save the data with tensorboard
    #----------------------------------------------------------------------------------------
        
    def saveTrainData(self):
        # Go through each scene
        for category, scene_list in self.init.dataset.items():     
            for scene in scene_list: 
                
                #~~~~~~~~~~~~~~~~~~~~~ Load dataset for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                print("~~~~~~~ Generating data ->>> " + category + " / " + scene + " ~~~~~~~~~~")

                dataset_dir = os.path.join(self.init.dataset_dir, category, scene, 'input')
                dataset_gt_dir = os.path.join(self.init.dataset_dir, category, scene, 'groundtruth')
                
                trainset = GenerateData(
                    dataset_gt_dir, dataset_dir,
                    framesBack=self.init.framesBack,
                    trainStart=self.init.trainStart,
                    trainEnd=self.init.trainEnd,
                    data_format=self.init.data_format,
                    void_value = False,
                    showSample = True
                )
                
                trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=self.init.batch_size, 
                    shuffle=self.init.shuffle, 
                    num_workers=self.init.num_workers
                )
                
                # Save to tensorboard
                tensorBoardTool = TensorBoardTool(self.init.logdir)
                idx = (self.init.trainEnd - self.init.trainStart)//2
                tensorBoardTool.saveDataloader(trainloader, idx)
                tensorBoardTool.saveNet(self.net, trainloader, self.device)
                tensorBoardTool.run()
                
        
    #----------------------------------------------------------------------------------------
    # Function to print a summary of the network
    #----------------------------------------------------------------------------------------
    
    def summaryNet(self):
        print(self.net)
        
        

