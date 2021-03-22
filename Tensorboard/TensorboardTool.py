import logging
import torch
import torchvision
import numpy      as np
import tensorflow as tf
from tensorboard              import default
from tensorboard              import program
from torch.utils.tensorboard  import SummaryWriter

class TensorBoardTool:

    def __init__(self, logdir, file_writer = None):
        self.logdir = logdir
        self.file_writer = file_writer
        
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
        
        # (b, c, t, h, w) -> (t, b, h, w, c)
        input_tensor = input.permute(2, 0, 3, 4, 1)[0]
        gt_tensor = gt_tensor.permute(2, 0, 3, 4, 1)[0]
        
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
    # Use tensorboard to save sample data
    #----------------------------------------------------------------------------------------
    
    def saveTrainLoss(self, name_folder, tag, loss, step):    
        if(self.file_writer == None):
            self.file_writer = SummaryWriter(self.logdir + '/' + name_folder)
            
        self.file_writer.add_scalar( tag, loss, step)
        

    def closeWriter(self):
        self.file_writer.close()
