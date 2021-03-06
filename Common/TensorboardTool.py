import logging
import torch
import torchvision
import numpy      as np
import tensorflow as tf
from tensorboard              import default
from tensorboard              import program
from torch.utils.tensorboard  import SummaryWriter

class TensorBoardTool:

    def __init__(self, logdir, dim5D, file_writer = None):
        self.logdir = logdir
        self.file_writer = file_writer
        self.dim5D  = dim5D
        
        
    #----------------------------------------------------------------------------------------
    # Run - Start tensorboard server
    # or in console run -> tensorboard --logdir=logs/train_data/
    #----------------------------------------------------------------------------------------

    def run(self):
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        tb = program.TensorBoard(plugins=default.get_plugins())
        tb.configure(argv=[None, '--logdir', self.logdir])
        url = tb.launch()
        print('TensorBoard at %s \n' % url)
        
        
    #----------------------------------------------------------------------------------------
    # Use tensorboard to save sample data
    #----------------------------------------------------------------------------------------
    
    def saveDataloader(self, trainloader, idx=0): 
        for i_batch, sample  in enumerate(trainloader):
            if(i_batch == idx):
                break
            
        input, gt_tensor = sample['inputs'], sample['gt']

        if (self.dim5D):    
            # (b, c, t, h, w) -> (t, b, h, w, c)
            input_tensor = input.permute (2, 0, 3, 4, 1)[0]
            gt_tensor = gt_tensor.permute(2, 0, 3, 4, 1)[0]
        else:
            # (b, c, h, w) -> (b, h, w, c)
            input_tensor = input.permute (0, 2, 3, 1)
            gt_tensor = gt_tensor.permute(0, 2, 3, 1)

        # normalize image
        input_tensor = torch.div(input_tensor, 255)
        gt_tensor = torch.div(gt_tensor, 255)
        
        file_writer = SummaryWriter(self.logdir + '/Sample_data')
        file_writer.add_images("Groundtruth images", gt_tensor,  dataformats='NHWC')
        file_writer.add_images("Input images", input_tensor,  dataformats='NHWC')
        file_writer.close()
        
        
    #----------------------------------------------------------------------------------------
    # Use tensorboard to save graph of the model
    #----------------------------------------------------------------------------------------
    
    def saveNet(self, net, trainloader, device): 
        # Obtain shape of input
        i, sample    = next(iter(enumerate(trainloader)))
        input_tensor = sample['inputs'].to(device)
        
        file_writer  = SummaryWriter(self.logdir + '/Model')
        file_writer.add_graph(net, input_to_model=input_tensor, verbose=False)
        file_writer.close()


    #----------------------------------------------------------------------------------------
    # Use tensorboard to add scalars
    #----------------------------------------------------------------------------------------
    
    def add_scalar(self, name_folder, tag, loss, step, comment=''):    
        if(self.file_writer == None):
            self.file_writer = SummaryWriter(self.logdir + '/' + name_folder, comment = comment)
            
        self.file_writer.add_scalar( tag, loss, step)
        

    def closeWriter(self):
        self.file_writer.close()


    #----------------------------------------------------------------------------------------
    # Plot img test
    #----------------------------------------------------------------------------------------
    
    def saveImgTest(self, model_name, i_step, inputs_, groundtruth_, prediction_):     
        if(self.dim5D):
            # (b, c, t, h, w) -> (t, b, c, h, w) -> get first frame of the sequence of frames
            inputs      = inputs_     .permute(2, 0, 1, 3, 4)[0]
            groundtruth = groundtruth_.permute(2, 0, 1, 3, 4)[0]
            prediction  = prediction_ .permute(2, 0, 1, 3, 4)[0]
        else :
            # (b, c, h, w) -> (b, h, w, c)
            inputs      = inputs_ 
            groundtruth = groundtruth_
            prediction  = prediction_

        # get only the first element of the batch
        inputs          = inputs[0]     .cpu().numpy() / 255.0
        groundtruth     = groundtruth[0].cpu().numpy() 
        prediction      = prediction[0] .cpu().numpy()
 
        # change -1 to a gray color
        shape = groundtruth.shape
        groundtruth     = groundtruth.reshape(-1)
        idx             = np.where(groundtruth==-1)[0] # find non-ROI
        if (len(idx)>0):
            groundtruth[idx] = 0.55
        groundtruth     = groundtruth.reshape(shape)
        
        
        inputs          = torch.from_numpy(inputs)
        groundtruth     = torch.from_numpy(groundtruth)
        prediction      = torch.from_numpy(prediction)
        
        # gray to rgb to concat three images in the future
        groundtruth = groundtruth.repeat(3, 1, 1)
        prediction  = prediction.repeat(3, 1, 1)
        
        # concat thre images into one image
        images = torch.cat((inputs, groundtruth, prediction), 2)

        # write on tensorboard
        file_writer = SummaryWriter(self.logdir + '/' )
        file_writer.add_images(model_name+'/'+str(i_step), images, global_step=None, dataformats='CHW')
        file_writer.close()
            
        