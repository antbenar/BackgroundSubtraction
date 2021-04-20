import os
import torch
import torch.nn              as nn
import torch.nn.functional   as F
import numpy                 as np
from Dataset.generateData                import GenerateData
from Common.TensorboardTool              import TensorBoardTool
from Common.Util                         import Averager
from Common.Util                         import ModelSize
from torch.utils.tensorboard             import SummaryWriter
from matplotlib                          import pyplot as plt
from tqdm                                import tqdm

class ModelTrainTest(nn.Module):
    def __init__(self, net, settings):
        super().__init__()
        self.net        = net
        self.settings = settings
        self._state     = {}
        
        # set settings to the model
        self._setSettings(settings)
        
        # net to device
        self.net.to(self.device)
        
        # optimizer
        self.optimizer = torch.optim.Adam(
                                self.net.parameters(),
                                lr=self.lr,
                                betas=(self.beta_a, self.beta_b)
                            )
        # scheduler
        if(self.scheduler_active):
            self.scheduler = torch.optim.lr_scheduler.StepLR( 
                                self.optimizer,
                                step_size = self.lr_decay_steps,
                                gamma     = self.lr_decay_factor
                            )

        # set lost
        self.criterion_loss = self._bce_loss
        
         
    #----------------------------------------------------------------------------------------
    # Set settings
    #----------------------------------------------------------------------------------------
      
    def _setSettings(self, settings):
        # model static attributes
        self.model_name       = settings.model_name
        self.activation       = settings.activation
        self.n_channels       = settings.n_channels
        self.p_dropout        = settings.p_dropout
        self.device           = settings.device
        # optimizer
        self.lr               = settings.lr
        self.beta_a           = settings.beta_a
        self.beta_b           = settings.beta_b
        #scheduler
        self.scheduler_active = settings.scheduler_active
        self.lr_decay_steps   = settings.lr_decay_steps
        self.lr_decay_factor  = settings.lr_decay_factor
        # train
        self.train_result_dir = settings.train_result_dir
        self.threshold        = settings.threshold
        self.view_batch       = settings.view_batch
        self.batch_size       = settings.batch_size
        self.shuffle          = settings.shuffle
        self.num_workers      = settings.num_workers
        self.epochs           = settings.epochs
        # dataset
        self.framesBack       = settings.framesBack
        self.dataset          = settings.dataset
        self.dataset_dir      = settings.dataset_dir
        self.resize           = settings.resize
        self.width            = settings.width
        self.height           = settings.height
        self.dataset_range    = settings.dataset_range
        self.trainStart       = settings.trainStart
        self.trainEnd         = settings.trainEnd
        self.data_format      = settings.data_format
        self.train_split      = settings.train_split
        self.val_split        = settings.val_split
        self.dataset_fg_bg    = settings.dataset_fg_bg
        self.showSample       = settings.showSample
        self.differenceFrames = settings.differenceFrames
        # test
        self.plot_test        = settings.plot_test
        self.log_test         = settings.log_test
        self.loadPath         = settings.loadPath
        # tensorboard
        self.logdir           = settings.logdir
        
        
    #----------------------------------------------------------------------------------------
    # Training state functions
    #----------------------------------------------------------------------------------------
      
    def _state_reset(self):
        self._state = {}
    def _state_add(self,name,attr):
        self._state[name]=attr
    def _state_save(self,epoch):
        pathMod = os.path.join(self.train_result_dir, self.category)
        if not os.path.exists(pathMod):
            os.makedirs(pathMod)
        # Save model
        pathMod = os.path.join(pathMod, 'mdl_' + self.category + '_' + self.scene + str(epoch) + '.pth')
        torch.save( self._state, pathMod)
    
    
    #----------------------------------------------------------------------------------------
    # load model
    #----------------------------------------------------------------------------------------
    
    def load(self, path):
        # Load
        checkpoint = torch.load(path)
        self.net      .load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint[ 'optimizer'])
        if(self.scheduler_active):
            self.scheduler.load_state_dict(checkpoint[ 'scheduler'])
        print('Model loaded\n')
        
    #----------------------------------------------------------------------------------------
    # Threshold function
    #----------------------------------------------------------------------------------------
    
    def _threshold(self, img, thd):
        return torch.Tensor.float(img > thd)

    
    #----------------------------------------------------------------------------------------
    # Validation metrics
    #----------------------------------------------------------------------------------------
    
    def _metrics(self, prediction, groundtruth):
        epsilon      = 1e-7
        numDecimales = 3    
        
        if(self.framesBack > 0):
            # (b, c, t, h, w) -> (t, b, h, w, c) -> get first frame of the sequence of frames
            prediction  = prediction.permute(2, 0, 3, 4, 1)[0]
            groundtruth = groundtruth.permute(2, 0, 3, 4, 1)[0]
        else :
            # (b, c, h, w) -> (b, h, w, c)
            prediction  = prediction.permute(0, 2, 3, 1)
            groundtruth = groundtruth.permute(0, 2, 3, 1)

        FP = torch.sum((prediction == 1) & (groundtruth == 0)).data.cpu().numpy()
        FN = torch.sum((prediction == 0) & (groundtruth == 1)).data.cpu().numpy()
        TP = torch.sum((prediction == 1) & (groundtruth == 1)).data.cpu().numpy()
        TN = torch.sum((prediction == 0) & (groundtruth == 0)).data.cpu().numpy()
    
        FMEASURE     = 0
        if(TP != 0):
            FMEASURE = round( 2*TP/(2*TP+FP+FN+epsilon), numDecimales)
            
        PWC          = round( 100*(FN+FP)/(TP+FN+FP+TN), numDecimales)
            
        return FMEASURE, PWC

    
    #----------------------------------------------------------------------------------------
    # Loss function
    #----------------------------------------------------------------------------------------
    
    def _bce_loss(self, prediction_, groundtruth_):# (5,2,240,320)
        void_label  = -1.
        prediction  = torch.reshape(prediction_, (-1,))
        groundtruth = torch.reshape(groundtruth_, (-1,))
        
        mask        = torch.where(groundtruth == void_label, False, True) # [0,1,1,0]
        
        prediction  = torch.masked_select(prediction, mask)
        groundtruth = torch.masked_select(groundtruth, mask) 
        
        loss        = F.binary_cross_entropy(prediction, groundtruth, reduction='mean')
        return loss
        
    
    #----------------------------------------------------------------------------------------
    # Function to train the model
    #----------------------------------------------------------------------------------------
        
    def _train(self, epoch, train_loader):
        # loss
        running_loss = 0.0 
        lossTrain    = Averager()
        with tqdm(total=len(train_loader),leave=True, desc="Epoch %d/%d" % (epoch,self.epochs)) as pbar:
            for i_batch, sample_batched  in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs      = sample_batched['inputs'].to(self.device)
                groundtruth = sample_batched['gt'].to(self.device)
        
                # zero the parameter gradients
                self.optimizer.zero_grad()
                self.net.zero_grad()
                
                # forward + backward + optimize
                outputs           = self.net(inputs)
                loss              = self.criterion_loss(outputs, groundtruth)
                loss.requres_grad = True
                loss.backward()
                self.optimizer.step()
                
                # print statistics every num_stat_batches
                runtime_loss      = loss.item()
                running_loss     += runtime_loss
                view_batch        = self.view_batch
                
                if (i_batch+1) % view_batch == 0:  # print every 2000 mini-batches
                    running_loss = 0.0
                    pbar.set_postfix({'Train loss':  '%.4f'%lossTrain.mean})
                    pbar.refresh()
                    
                pbar.update()
                pbar.refresh()
                lossTrain.update(runtime_loss)
                del loss, outputs
                torch.cuda.empty_cache()
                
            pbar.close()
        lossTrain = lossTrain.val()
        return lossTrain
        
    
    #----------------------------------------------------------------------------------------
    # Function to validate the training of the model
    #----------------------------------------------------------------------------------------
        
    def _validation(self, epoch, val_loader):
        #loss
        running_loss = 0.0 
        lossVal      = Averager()
        
        # Metrics 
        avgFmeasure  = Averager()
        avgPWC       = Averager()
        
        with tqdm(total=len(val_loader),leave=True, desc="Epoch %d/%d" % (epoch,self.epochs)) as pbar:
            for i_batch, sample_batched  in enumerate(val_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs      = sample_batched['inputs'].to(self.device)
                groundtruth = sample_batched['gt'].to(self.device)
        
    
                # Predict
                outputs     = self.net(inputs)
                prediction  = self._threshold(outputs, self.threshold)
                loss        = self.criterion_loss(outputs, groundtruth)
                
                # Metrics
                fmeasure, pwc = self._metrics(prediction, groundtruth)
                avgFmeasure.update(fmeasure)  
                avgPWC.update(pwc)  
                
                # print statistics every num_stat_batches
                runtime_loss  = loss.item()
                running_loss += runtime_loss
                view_batch = self.view_batch
                
                if (i_batch+1) % view_batch == 0:  # print every 2000 mini-batches
                    running_loss = 0.0
                    pbar.set_postfix({'Val loss':  '%.4f'%lossVal.val(), 'f-measure':  '%.4f'%avgFmeasure.mean,'PWC':  '%.4f'%avgPWC.mean})
                    pbar.refresh()
                    
                pbar.update() 
                pbar.refresh()
                    
                lossVal.update(runtime_loss)
                del loss, outputs, inputs, groundtruth
                torch.cuda.empty_cache()
                
            pbar.close()
            
        lossVal     = lossVal.val()
        avgFmeasure = avgFmeasure.mean
        avgPWC = avgPWC.mean
        return lossVal, avgFmeasure, avgPWC
    
    
    #----------------------------------------------------------------------------------------
    # Train the model for each scene
    #----------------------------------------------------------------------------------------
    
    def _train_val(self, train_loader, val_loader):    
        category = self.category 
        scene    = self.scene
            
        print("\n~~~~~~~ Training ->>> " + category + " / " + scene + " ~~~~~~~~~~\n")
        tb = SummaryWriter(self.logdir +'/' + category + '_' + scene)
        
        for epoch in range(self.epochs):  # loop over the dataset multiple times            
            lossTrain                = self._train(epoch, train_loader)
            lossValid, fmeasure, pwc = self._validation(epoch, val_loader)
            
            if(self.scheduler_active):
                self.scheduler.step()

            # add scalars to tensorboard
            tb.add_scalar('Loss/Train'        ,   lossTrain, epoch)
            tb.add_scalar('Loss/Validation'   ,   lossValid, epoch)
            tb.add_scalar('Metrics/Val/F-Measure' , fmeasure, epoch)
            tb.add_scalar('Metrics/Val/PWC'       , pwc, epoch)
            
            # Save checkpoint
            if epoch%2 == 0 or epoch == (self.epochs -1):
                self._state_add (     'epoch',                    epoch  )
                self._state_add ('state_dict',self.      net.state_dict())
                self._state_add ( 'optimizer',self.optimizer.state_dict())
                if(self.scheduler_active):
                    self._state_add ( 'scheduler',self.scheduler.state_dict())
                self._state_save(epoch)
                
        
        tb.close()

        
    #----------------------------------------------------------------------------------------
    # Function to test the trained model
    #----------------------------------------------------------------------------------------
        
    def _test(self, test_loader):
        category = self.category 
        scene    = self.scene
            
        dir_tb = self.logdir + '/test/' + category + '_' + scene
        tb = SummaryWriter(dir_tb)
        
        # Metrics 
        avgFmeasure  = Averager()
        avgPWC       = Averager()
        step_view = len(test_loader)//5
        
        with tqdm(total=len(test_loader),leave=False,desc='Test') as pbar:
            for i_step, sample_batched  in enumerate(test_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs      = sample_batched['inputs'].to(self.device)
                groundtruth = sample_batched['gt'].to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.net(inputs)
    
                if(self.activation == 'softmax'):
                    prediction  = torch.argmax(outputs, dim=1, keepdim=True).float()
                else:
                    prediction  = self._threshold(outputs, self.threshold)
                
                # plot
                if(self.plot_test):
                    self.plotImgTest(i_step, inputs, groundtruth, prediction)
                #self.saveImgTest(i_step, inputs, groundtruth, prediction)
                
                # Metrics
                fmeasure, pwc = self._metrics(prediction, groundtruth)
                avgFmeasure.update(fmeasure)  
                avgPWC.update(pwc)  
                
    
                
                if (i_step+1)%step_view==0:  # print every x mini-batches
                    if self.log_test:
                        # add scalars to tensorboard
                        tb.add_scalar('Test/F-Measure' , avgFmeasure.mean, (i_step+1)//step_view)
                        tb.add_scalar('Test/PWC'       , avgPWC.mean     , (i_step+1)//step_view)
                        
                        # add segmentation and groundtruth to tensorboard
                        dim5D = self.framesBack > 0
                        tensorBoardTool = TensorBoardTool(dir_tb, dim5D)
                        tensorBoardTool.saveImgTest(self.model_name, (i_step+1)//step_view, inputs, groundtruth, prediction)
                        
                    pbar.set_postfix({'f-measure':  '%.4f'%avgFmeasure.mean,'PWC':  '%.4f'%avgPWC.mean})
                pbar.update()
                pbar.refresh()
                
                del outputs, inputs, groundtruth
                torch.cuda.empty_cache()
                
            pbar.close()
        
        print("Test metrics - f-measure:", avgFmeasure.mean, ", PWC:", avgPWC.mean)
        return avgFmeasure, avgPWC

        
    #----------------------------------------------------------------------------------------
    # Train the model for each scene
    #----------------------------------------------------------------------------------------
    
    def _execute(self, mode):
        # try:
        category = self.category 
        scene    = self.scene
        #~~~~~~~~~~~~~~~~~~~~~ Load dataset for this scene ~~~~~~~~~~~~~~~~~~~~~
        
        print("~~~~~~~ Generating data ->>> " + category + " / " + scene + " ~~~~~~~~~~")

        dataset_dir = os.path.join(self.dataset_dir, category, scene, 'input')
        dataset_gt_dir = os.path.join(self.dataset_dir, category, scene, 'groundtruth')
        
        dataset = GenerateData(
            dataset_gt_dir, dataset_dir,
            framesBack    = self.framesBack,
            resize        = self.resize,
            width         = self.width,
            height        = self.height,
            dataset_range = self.dataset_range,
            trainStart    = self.trainStart,
            trainEnd      = self.trainEnd,
            data_format   = self.data_format,
            dataset_fg_bg = self.dataset_fg_bg,
            showSample    = self.showSample,
            differenceFrames = self.differenceFrames
        )
        
        train_loader, val_loader, test_loader = dataset.train_val_test_split(
                                                    self.train_split,
                                                    self.val_split,
                                                    self.shuffle,
                                                    self.batch_size,
                                                    self.num_workers
                                                  )
        
        if(mode == 'train_val'):
            self._train_val(train_loader, val_loader)
        elif(mode == 'train_val_test'):
            self._train_val(train_loader, val_loader)
            self._test(test_loader)
        elif(mode == 'test'):
            loadpath = os.path.join(self.loadPath, category, 'mdl_'+category+'_'+scene+'49.pth')
            self.load(loadpath)
            self._test(test_loader)
        else: 
            raise NameError('Mode (' + mode + ') not valid')
        
        del dataset, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        
        
        
        
       
        # except RuntimeError as err:
        #     if(self.batch_size > 1):
        #         self.batch_size -= 1
        #         print('===============reducing batch size to:', self.batch_size)                  
        #         return  self._execute()
        #     else:
        #         print(colored('='*20, 'red'))
        #         print(colored('RuntimeError', 'red'), err)
                  
                
    #----------------------------------------------------------------------------------------
    # Function to iterate over categories and scenes and train the model for each one
    #----------------------------------------------------------------------------------------
    
    def execute(self, mode):
        
        print('~~~~~~~~~~~~~~ Current method >>> ' + self.model_name)
                
        # Go through each scene
        for category, scene_list in self.dataset.items():     
            for scene in scene_list:
                self.category = category
                self.scene    = scene
                
                try:
                    self._execute(mode)
                except ValueError as err:
                    print("Error: ", err)
                    
                
                
    #----------------------------------------------------------------------------------------
    # Function to iterate over categories and scenes to save the data with tensorboard
    #----------------------------------------------------------------------------------------
        
    def saveDataTensorboard(self):
        # Go through each scene
        for category, scene_list in self.dataset.items():     
            for scene in scene_list: 
                
                #~~~~~~~~~~~~~~~~~~~~~ Load dataset for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                print("~~~~~~~ Generating data ->>> " + category + " / " + scene + " ~~~~~~~~~~")

                dataset_dir = os.path.join(self.dataset_dir, category, scene, 'input')
                dataset_gt_dir = os.path.join(self.dataset_dir, category, scene, 'groundtruth')
                
                dataset = GenerateData(
                    dataset_gt_dir, dataset_dir,
                    framesBack    = self.framesBack,
                    resize        = self.resize,
                    width         = self.width,
                    height        = self.height,
                    dataset_range = self.dataset_range,
                    trainStart    = self.trainStart,
                    trainEnd      = self.trainEnd,
                    data_format   = self.data_format,
                    void_value    = False,
                    showSample    = self.showSample
                )
                
                dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=self.shuffle, 
                    num_workers=self.num_workers
                )
                
                # Save to tensorboard
                dim5D = self.framesBack > 0
                tensorBoardTool = TensorBoardTool(self.logdir, dim5D)
                idx = len(dataloader)//2
                tensorBoardTool.saveDataloader(dataloader, idx)
                tensorBoardTool.saveNet(self.net, dataloader, self.device)
                #tensorBoardTool.run()
              
                
    #----------------------------------------------------------------------------------------
    # Function to calculate memory on the model
    #----------------------------------------------------------------------------------------
        
    def calculateModelSize(self):
        # Go through each scene
        for category, scene_list in self.dataset.items():     
            for scene in scene_list: 
                
                #~~~~~~~~~~~~~~~~~~~~~ Load dataset for this scene ~~~~~~~~~~~~~~~~~~~~~
                
                print("~~~~~~~ Generating data ->>> " + category + " / " + scene + " ~~~~~~~~~~")

                dataset_dir = os.path.join(self.dataset_dir, category, scene, 'input')
                dataset_gt_dir = os.path.join(self.dataset_dir, category, scene, 'groundtruth')
                
                dataset = GenerateData(
                    dataset_gt_dir, dataset_dir,
                    framesBack    = self.framesBack,
                    resize        = self.resize,
                    width         = self.width,
                    height        = self.height,
                    dataset_range = self.dataset_range,
                    trainStart    = self.trainStart,
                    trainEnd      = self.trainEnd,
                    data_format   = self.data_format,
                    void_value    = False,
                    showSample    = self.showSample
                )
                
                dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=self.shuffle, 
                    num_workers=self.num_workers
                )
                _, sample_batched = next(iter(enumerate(dataloader)))
                inputs      = sample_batched['inputs']
                ModelSize(inputs, self.net, self.device)
                
                del dataset, dataloader, inputs, self.net
                torch.cuda.empty_cache()

                
    #----------------------------------------------------------------------------------------
    # Function to print a summary of the network
    #----------------------------------------------------------------------------------------
    
    def summaryNet(self):
        print(self.net)
        
        
    #----------------------------------------------------------------------------------------
    # Plot img test
    #----------------------------------------------------------------------------------------
    
    def plotImgTest(self, i_step, inputs_, groundtruth_, prediction_):     
        
        if(self.framesBack > 0):
            # (b, c, t, h, w) -> (t, b, h, w, c) -> get first frame of the sequence of frames
            inputs      = inputs_.permute(2, 0, 3, 4, 1)[0]
            groundtruth = groundtruth_.permute(2, 0, 3, 4, 1)[0]
            prediction  = prediction_.permute(2, 0, 3, 4, 1)[0]
        else :
            # (b, c, h, w) -> (b, h, w, c)
            inputs      = inputs_.permute(0, 2, 3, 1)
            groundtruth = groundtruth_.permute(0, 2, 3, 1)
            prediction  = prediction_.permute(0, 2, 3, 1)

        # get only the first element of the batch
        inputs      = inputs[0].cpu().numpy()
        groundtruth = groundtruth[0].cpu().numpy()
        prediction  =  prediction[0].cpu().numpy()
 
        # change -1 to a gray color
        shape          = groundtruth.shape
        groundtruth    = groundtruth.reshape(-1)
        idx            = np.where(groundtruth==-1)[0] # find non-ROI
        if (len(idx)>0):
            groundtruth[idx] = 0.55
        groundtruth = groundtruth.reshape(shape)
            
        ax = plt.subplot(1, 3, 1)
        ax.set_title('Input #{}'.format(i_step))
        ax.axis('off')
        plt.imshow(inputs/255)
        plt.tight_layout()
        
        ax2 = plt.subplot(1, 3, 2)
        ax2.set_title('Gt #{}'.format(i_step))
        ax2.axis('off')
        plt.imshow(groundtruth, cmap=plt.get_cmap('gray'))
        plt.tight_layout()

        ax3 = plt.subplot(1, 3, 3)
        ax3.set_title('Predict #{}'.format(i_step))
        ax3.axis('off')
        plt.imshow(prediction, cmap=plt.get_cmap('gray'))
        plt.tight_layout()
        
        plt.show()
        
    #----------------------------------------------------------------------------------------
    # save img test
    #----------------------------------------------------------------------------------------
    
    def saveImgTest(self, step, inputs_, groundtruth_, prediction_):     
        
        if(self.framesBack > 0):
            # (b, c, t, h, w) -> (t, b, h, w, c) -> get first frame of the sequence of frames
            inputs      = inputs_.permute(2, 0, 3, 4, 1)[0]
            groundtruth = groundtruth_.permute(2, 0, 3, 4, 1)[0]
            prediction  = prediction_.permute(2, 0, 3, 4, 1)[0]
        else :
            # (b, c, h, w) -> (b, h, w, c)
            inputs      = inputs_.permute(0, 2, 3, 1)
            groundtruth = groundtruth_.permute(0, 2, 3, 1)
            prediction  = prediction_.permute(0, 2, 3, 1)

        # get only the first element of the batch
        inputs      = inputs[0].cpu().numpy()
        groundtruth = groundtruth[0].cpu().numpy()
        prediction  =  prediction[0].cpu().numpy()
 
        # change -1 to a gray color
        shape          = groundtruth.shape
        groundtruth    = groundtruth.reshape(-1)
        idx            = np.where(groundtruth==-1)[0] # find non-ROI
        if (len(idx)>0):
            groundtruth[idx] = 0.55
        groundtruth = groundtruth.reshape(shape)
 
        #save img
        pathTest = os.path.join('TestResult', self.model_name, self.category, self.scene)
        if not os.path.exists(pathTest):
            os.makedirs(pathTest)
            
            
        pathTest = os.path.join(pathTest, str(step))
        
        fig, (ax, ax2, ax3) = plt.subplots(1, 3)
        ax.set_title('Input #{}'.format(step))
        ax.axis('off')
        ax.imshow(inputs/255)
        plt.tight_layout()
        
        ax2.set_title('Gt #{}'.format(step))
        ax2.axis('off')
        ax2.imshow(groundtruth, cmap=plt.get_cmap('gray'))
        plt.tight_layout()

        ax3.set_title('Predict #{}'.format(step))
        ax3.axis('off')
        ax3.imshow(prediction, cmap=plt.get_cmap('gray'))
        plt.tight_layout()

        plt.savefig(pathTest)
        
        plt.close('all')
