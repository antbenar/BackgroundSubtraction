import os
import torch
import torch.nn as nn
import torch.optim as optim
from Model.model import Net
from Dataset.generateData import GenerateData

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
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.init.lr, betas=(0.9, 0.999))
        
        for epoch in range(self.init.epochs):  # loop over the dataset multiple times
            print("len_trainloader = ",len(trainloader))
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
        
                # print statistics
                running_loss += loss.item()
                if i_batch % 2 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i_batch + 1, running_loss / 2000))
                    running_loss = 0.0
                    
                del loss, outputs
                
        print('Finished Training')
        
        
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
                    data_format=self.init.data_format
                )
                
                trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=self.init.batch_size, 
                    shuffle=self.init.shuffle, 
                    num_workers=self.init.num_workers
                )
                
                ## plot the intermediate frame to qualitatively validate our trainset
                # trainset.plotSample(idx_frame=(self.init.trainEnd-self.init.trainStart)//2)
                
                #~~~~~~~~~~~~~~~~~~~~~ Train net for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                #print("~~~~~~~ Training ->>> " + category + " / " + scene + " ~~~~~~~~~~")
                train_result_dir = os.path.join(self.init.train_result_dir, 'mdl_' + category + '_' + scene + '.h5')
                self.train(trainloader, train_result_dir)
                del trainset
          
                
    #----------------------------------------------------------------------------------------
    # Function to print a summary of the network
    #----------------------------------------------------------------------------------------
    
    def summaryNet(self):
        print(self.net)
        
        
    #----------------------------------------------------------------------------------------
    # Use tensorboard to save data of the model
    #----------------------------------------------------------------------------------------
    """
    def saveTensorboard(self, idx_frame): 
        
        inputs, gt = self.dataset[0][idx_frame], self.dataset[1][idx_frame]
        
        if (self.data_format=='channels_last'):
            #Given a sequence of frames, divide in groups of five consecutive frames
            inputs = np.moveaxis(inputs, 0, -1)
            gt = np.moveaxis(gt, 0, -1)
        
        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter()
        grid = torchvision.utils.make_grid(images)
        writer.add_image('Input #{}'.format(idx_frame), grid, 0)
        writer.add_graph(model, images)
        writer.close()
    """
