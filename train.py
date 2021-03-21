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
        
        
    def train(self, trainloader, train_result_dir):
        
        criterion_loss = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.init.lr, betas=(0.9, 0.999))
        
        for epoch in range(self.init.epochs):  # loop over the dataset multiple times
            print("len_trainloader = ",len(trainloader))
            running_loss = 0.0
            
            for i_batch, sample_batched  in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                print("Values batch ", i_batch, sample_batched['inputs'].size(), sample_batched['gt'].size())


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
                if i_batch % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i_batch + 1, running_loss / 2000))
                    running_loss = 0.0
                
        print('Finished Training')
        
        # save the model
        torch.save(self.net.state_dict(), train_result_dir)
        
        
        
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
                trainset.plotSample(idx_frame=(self.init.trainEnd-self.init.trainStart)//2)
                
                #~~~~~~~~~~~~~~~~~~~~~ Train net for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                #print("~~~~~~~ Training ->>> " + category + " / " + scene + " ~~~~~~~~~~")
                train_result_dir = os.path.join(self.init.train_result_dir, 'mdl_' + category + '_' + scene + '.h5')
                self.train(trainloader, train_result_dir)
                del trainset
                
        
    def summaryNet(self):
        print(self.net)
