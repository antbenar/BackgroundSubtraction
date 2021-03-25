import os
import torch
import torch.nn              as nn
import numpy                 as np
import torch.nn.functional   as F
from Model.Base_model                    import Net
#from Model.Base2D.Base2D_model           import Net
#from Model.Unet2D.Unet2D_model           import Net
from Dataset.generateData                import GenerateData
from Common.TensorboardTool              import TensorBoardTool
from Common.Util                         import Averager
from Common.Util                         import ModelSize
from torch.utils.data.sampler            import SubsetRandomSampler
from torch.utils.tensorboard             import SummaryWriter


class ModelTrain(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.init = init
        
        # model static attributes
        self.n_channels = init.n_channels
        self.p_dropout = init.p_dropout
        self.device = init.device
        self._state    = {}
        
        # instance the model
        self.net = Net(
            self.n_channels, 
            self.p_dropout, 
            up_mode= self.init.up_mode
        )
        # net to device
        self.net.to(self.device)
        
        #optimizer
        self.optimizer    = torch.optim.Adam(
                                self.net.parameters(),
                                lr=self.init.lr,
                                betas=(self.init.beta_a, self.init.beta_b)
                            )

        # set lost
        self.criterion_loss = self._bce_loss
        
        
    #----------------------------------------------------------------------------------------
    # Training state functions
    #----------------------------------------------------------------------------------------
      
    def _state_reset(self):
        self._state = {}
    def _state_add(self,name,attr):
        self._state[name]=attr
    def _state_save(self,epoch):
        # Save model
        pathMod = os.path.join(self.init.train_result_dir, 'mdl_' + self.category + '_' + self.scene + str(epoch) + '.pth')
        torch.save( self._state, pathMod)
     
    #----------------------------------------------------------------------------------------
    # Loss function
    #----------------------------------------------------------------------------------------
    def _bce_loss(self, prediction_, groundtruth_):
        void_label  = -1.
        prediction  = torch.reshape(prediction_, (-1,))
        groundtruth = torch.reshape(groundtruth_, (-1,))
        
        mask        = torch.where(groundtruth == void_label, False, True)
        
        prediction  = torch.masked_select(prediction, mask)
        groundtruth = torch.masked_select(groundtruth, mask)

        loss        = F.binary_cross_entropy(prediction, groundtruth, reduction='mean')
        return loss

    #----------------------------------------------------------------------------------------
    # Threshold function
    #----------------------------------------------------------------------------------------
    def _threshold(self, img, thd):
        img[img >= thd] = 1
        img[img < thd]  = 0
        return img  
    
    
    #----------------------------------------------------------------------------------------
    # Validation metrics
    #----------------------------------------------------------------------------------------

    def _metrics(self, prediction, groundtruth):
        if(self.init.framesBack > 0):
            # (b, c, t, h, w) -> (t, b, h, w, c) -> get first frame of the sequence of frames
            prediction  = prediction.permute(2, 0, 3, 4, 1)[0]
            groundtruth = groundtruth.permute(2, 0, 3, 4, 1)[0]
        else :
            # (b, c, h, w) -> (b, h, w, c)
            prediction  = prediction.permute(0, 2, 3, 1)
            groundtruth = groundtruth.permute(0, 2, 3, 1)
        
        
        prediction  = self._threshold(prediction, self.init.threshold)

        FP = torch.sum((prediction == 1) & (groundtruth == 0)).data.cpu().numpy()
        FN = torch.sum((prediction == 0) & (groundtruth == 1)).data.cpu().numpy()
        TP = torch.sum((prediction == 1) & (groundtruth == 1)).data.cpu().numpy()
        TN = torch.sum((prediction == 0) & (groundtruth == 0)).data.cpu().numpy()

        numDecimales = 3    

        FMEASURE     = round( 2*TP/(2*TP+FP+FN), numDecimales)
        PWC          = round( 100*(FN+FP)/(TP+FN+FP+TN), numDecimales)
        
        metricsMean  = [FMEASURE, PWC]

        return metricsMean
        
    #----------------------------------------------------------------------------------------
    # Function to train the model
    #----------------------------------------------------------------------------------------
        
    def _train(self, epoch, train_loader):
        # loss
        running_loss = 0.0 
        lossTrain    = Averager()
        
        for i_batch, sample_batched  in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs      = sample_batched['inputs'].to(self.device)
            groundtruth = sample_batched['gt'].to(self.device)
    
            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.net .zero_grad()
            
            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion_loss(outputs, groundtruth)
            loss.backward()
            self.optimizer.step()
            
            # print statistics every num_stat_batches
            runtime_loss = loss.item()
            running_loss += runtime_loss
            view_batch = self.init.view_batch
            
            if (i_batch+1) % view_batch == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / view_batch))
                running_loss = 0.0
                
            lossTrain.update(runtime_loss)
            del loss, outputs
            
        lossTrain = lossTrain.val()
        print("Epoch training loss:",lossTrain)
        return lossTrain
        
    #----------------------------------------------------------------------------------------
    # Function to validate the training of the model
    #----------------------------------------------------------------------------------------
        
    def _validation(self, epoch, val_loader):
        #loss
        running_loss = 0.0 
        lossVal      = Averager()
        
        # Metrics [Steer,Gas,Brake]
        avgMetrics   = Averager(2)
        
        for i_batch, sample_batched  in enumerate(val_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs      = sample_batched['inputs'].to(self.device)
            groundtruth = sample_batched['gt'].to(self.device)
    

            # Predict
            outputs = self.net(inputs)
            loss    = self.criterion_loss(outputs, groundtruth)

            # Metrics
            mean    = self._metrics(outputs, groundtruth)
            avgMetrics.update(mean)  
            
            # print statistics every num_stat_batches
            runtime_loss  = loss.item()
            running_loss += runtime_loss
            view_batch = self.init.view_batch
            
            if (i_batch+1) % view_batch == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / view_batch))
                running_loss = 0.0
                
            lossVal.update(runtime_loss)
            del loss, outputs
            
        lossVal = lossVal.val()
        avgMetrics = avgMetrics.mean
        
        print("Epoch val train loss:",lossVal, ", f-measure:", avgMetrics[0], ", PWC:", avgMetrics[1])
        return lossVal, avgMetrics
    
        
        
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
                self.category = category
                self.scene    = scene

                #~~~~~~~~~~~~~~~~~~~~~ Load dataset for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                print("~~~~~~~ Generating data ->>> " + category + " / " + scene + " ~~~~~~~~~~")

                dataset_dir = os.path.join(self.init.dataset_dir, category, scene, 'input')
                dataset_gt_dir = os.path.join(self.init.dataset_dir, category, scene, 'groundtruth')
                
                dataset = GenerateData(
                    dataset_gt_dir, dataset_dir,
                    framesBack  = self.init.framesBack,
                    trainStart  = self.init.trainStart,
                    trainEnd    = self.init.trainEnd,
                    data_format = self.init.data_format,
                    resize      = self.init.resize,
                    showSample  = True
                )
                
                train_loader, val_loader, test_loader = self.train_val_test_split(dataset)
                del test_loader # test_loader dont used here
                
                #~~~~~~~~~~~~~~~~~~~~~ Train net for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                print("~~~~~~~ Training ->>> " + category + " / " + scene + " ~~~~~~~~~~")
                #TensorBoardTool(self.init.logdir).run()
                tb = SummaryWriter(self.init.logdir)
                
                for epoch in range(self.init.epochs):  # loop over the dataset multiple times
                    print("Epoch = ", epoch, "-"*40)
                    
                    lossTrain       =      self._train(epoch, train_loader)
                    lossValid, metr = self._validation(epoch, val_loader)

                    # add scalars to tensorboard
                    tb.add_scalar('Loss/Train'        , lossTrain, epoch)
                    tb.add_scalar('Loss/Validation'   , lossValid, epoch)
                    tb.add_scalar('Metrics/F-Measure' , metr[0]  , epoch)
                    tb.add_scalar('Metrics/PWC'       , metr[1]  , epoch)
                    
                    # Save checkpoint
                    if epoch%2 == 0 or epoch == (self.init.epochs -1):
                        self._state_add (     'epoch',                    epoch  )
                        self._state_add ('state_dict',self.      net.state_dict())
                        self._state_add ( 'optimizer',self.optimizer.state_dict())
                        self._state_save(epoch)
                
                tb.close()
                del dataset, train_loader, val_loader
          
            
    #----------------------------------------------------------------------------------------
    # Function to separate the whole dataset into three subsets for training, validation of training and test
    #----------------------------------------------------------------------------------------
        
    def train_val_test_split(self, dataset):
        train_split_  = self.init.train_split
        val_split_    = self.init.val_split
        shuffle       = self.init.shuffle
        
        dataset_size = len(dataset)
        indices      = list(range(dataset_size))
        train_split  = int(np.floor(train_split_ * dataset_size))                # obtain the number of train samples
        val_split    = int(np.floor(train_split_ * val_split_ * dataset_size))   # obtain the number of val samples
        train_split  = train_split - val_split                                   # obtain the position where the train samples end
        val_split    = train_split + 2*val_split                                 # obtain the position where the val samples end
        
        if shuffle :
            np.random.seed(1234)
            np.random.shuffle(indices)
            
        train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

        # Creating PT data samplers and loaders:
        train_sampler  = SubsetRandomSampler(train_indices)
        valid_sampler  = SubsetRandomSampler(val_indices)
        test_sampler   = SubsetRandomSampler(test_indices)
                
        train_loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size  = self.init.batch_size, 
                            shuffle     = self.init.shuffle, 
                            num_workers = self.init.num_workers,
                            sampler     = train_sampler
                        )
        
        val_loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size  = self.init.batch_size, 
                            shuffle     = self.init.shuffle, 
                            num_workers = self.init.num_workers,
                            sampler     = valid_sampler
                        )
        
        test_loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size  = self.init.batch_size, 
                            shuffle     = self.init.shuffle, 
                            num_workers = self.init.num_workers,
                            sampler     = test_sampler
                        )

        return train_loader, val_loader, test_loader
    
    
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
                    resize      = self.init.resize,
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
                idx = len(trainloader)//2
                tensorBoardTool.saveDataloader(trainloader, idx)
                tensorBoardTool.saveNet(self.net, trainloader, self.device)
                #tensorBoardTool.run()
                
    #----------------------------------------------------------------------------------------
    # Function to calculate memory on the model
    #----------------------------------------------------------------------------------------
        
    def calculateModelSize(self):
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
                    resize      = self.init.resize,
                    void_value = False,
                    showSample = True
                )
                
                trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=self.init.batch_size, 
                    shuffle=self.init.shuffle, 
                    num_workers=self.init.num_workers
                )
                _, sample_batched = next(iter(enumerate(trainloader)))
                inputs      = sample_batched['inputs']
                ModelSize(inputs, self.net, self.device)
                
                return;

                
    #----------------------------------------------------------------------------------------
    # Function to print a summary of the network
    #----------------------------------------------------------------------------------------
    
    def summaryNet(self):
        print(self.net)
        
        

