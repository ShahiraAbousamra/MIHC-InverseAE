import sys;
import os;
import numpy as np;

from ..sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch;

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob;
import torchvision.models;
from distutils.util import strtobool;
from collections.abc import Iterable;



class MultiplexAutoencoderFixedStainsArch3Next3InputStain(nn.Module):
    def __init__(self, n_channels, model_out_path, model_base_filename, model_restore_filename, cost_func, device, kwargs):
        super(MultiplexAutoencoderFixedStainsArch3Next3InputStain, self).__init__();
        self.n_channels = n_channels;
        self.model_out_path = model_out_path;
        self.model_base_filename = model_base_filename;
        self.model_restore_filename = model_restore_filename;
        self.current_model_checkpoint_path = None;
        self.cost_func = cost_func;
        self.device = device;

        self.stains_dict = {};
        stains = np.array([[151, 82, 62], [118, 62, 151], [30, 30, 30], [62, 147, 151], [165, 168, 45], [174, 38, 75], [62, 104, 151], [221, 220, 219]]).transpose();
        self.stains_dict['shahira_wsi2'] = stains;
        self.stains_dict['shahira_wsi2-inv'] = 255 - stains;

        self.create_model(kwargs);



    def create_model(self, kwargs):
        # predefined list of arguments
        args = {'stain_init_name': 'shahira_wsi2'
            , 'conv_init': 'uniform', 'use_softmax':'True', 'n_channels':3
            , 'dropout_keep_prob' : 1.0, 'softmax_temp': 1.0
            , 'device' : torch.device("cpu")
        };

        args.update(kwargs);

        # 'conv_init': 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'he'

        # read extra argument
        self.stain_init_name = str(args['stain_init_name']);
        self.conv_init = str(args['conv_init']).lower();
        self.use_softmax = bool(strtobool(args['use_softmax']));
        self.softmax_temp = float(args['softmax_temp']);
        self.n_channels = int(args['n_channels']);
        self.n_stains = self.stains_dict[self.stain_init_name].shape[1];
        self.dropout_keep_prob = float(args['dropout_keep_prob'])


        self.encoder = nn.Sequential(                    
            nn.Conv2d(self.n_channels+3, 128, kernel_size=1),  
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3), 
            nn.MaxPool2d(2, stride=2),                   
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True), 
            nn.Conv2d(64, 32, kernel_size=3),            
            nn.MaxPool2d(2, stride=2),                   
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3),           
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True));

    
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16, 16, stride=2, kernel_size=3),  
            nn.Conv2d(16, 32, kernel_size=3),                     
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, stride=2, kernel_size=3),  
            nn.Conv2d(32, 64, kernel_size=3),                     
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3),                    
            nn.Dropout(p=self.dropout_keep_prob), 
            nn.ReLU(True),
            nn.Conv2d(128, 1, kernel_size=1),         
            );

        self.init_layers(self.encoder);
        self.init_layers(self.decoder);

        self.softmax_layer = torch.nn.Softmax(dim=1);


        init_stains_rgb = self.stains_dict[self.stain_init_name];
        self.rgb_stains = init_stains_rgb.T;

        init_stains_od = self.transform_intensity_to_optical_density(init_stains_rgb);
        self.od_stains_tensor = torch.from_numpy(init_stains_od).float();
        self.od_stains = nn.Parameter(data=self.od_stains_tensor, requires_grad=False);
        print('self.od_stains = ', self.od_stains);

        self.zero_grad() ;


        encoder_params_count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params_count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print('encoder_params_count', encoder_params_count)
        print('decoder_params_count', decoder_params_count)
        print('sum_params_count', decoder_params_count+encoder_params_count)



    def forward(self, x):
        self.stains_channel = [];
        conc_list = [];
        print('x', x.shape)
        # Loop over stains
        for s in range (self.rgb_stains.shape[0]):
            stain_val = self.rgb_stains[s];
            # Create conditional input channels that represent an RGB image with all pixels have the color of the conditional reference stain. 
            stain_channel = np.ones((x.shape[0] ,3, x.shape[2], x.shape[3])) * stain_val.reshape((1,3,1,1));
            stain_channel = torch.tensor(stain_channel, dtype = torch.float).to(self.device)
            # Concatenate the conditional input to the batch images
            input_w_stain = torch.cat((x, stain_channel), 1);
            # Pass new input to the model
            enc_s = self.encoder(input_w_stain);
            dec_s = self.decoder(enc_s);
            # Collect the current stain prediction
            conc_list.append(dec_s)
        # concatenate all the stains outputs
        c = torch.cat(conc_list, 1)  
            
        # square the predicted stain concentration maps
        c1 = c**2; # use the squares  to avoid negative values
        if(self.use_softmax):
            c1 = self.softmax_layer(c1/self.softmax_temp)
        c2 = c1.view(-1, self.n_stains, c1.size()[2]*c1.size()[3]); # flatten the stains before multiply
        # Create the stain maps in the OD space
        o = torch.matmul(self.od_stains, c2);
        return o.view(-1, self.n_channels, c.size()[2], c.size()[3]), c1, c2.view(-1, self.n_stains, c.size()[2], c.size()[3]), self.od_stains;


    def get_prediction_softmax(self, logits):
        return CNNArchUtils.get_probability_softmax(logits);

    def get_class_prediction(self, logits):
        probability, predicted_class = torch.max(logits.data, 1)
        return predicted_class ;

    def get_correct_prediction(self, logits, labels):
        prediction = self.get_class_prediction(logits);
        return (torch.eq(prediction, labels));


    def calc_out_size_conv2d(self, in_width, in_height, kernel_size):
        out_width = -1;
        out_height = -1;
        if(in_width > 0):
            out_width = in_width - kernel_size + 1;
        if(in_height > 0):
            out_height = in_height - kernel_size + 1;
        return out_width, out_height;

    def calc_out_size_maxpool2d(self, in_width, in_height, kernel_size):
        out_width = -1;
        out_height = -1;
        if(in_width > 0):
            out_width = in_width//2;
        if(in_height > 0):
            out_height = in_height//2;
        return out_width, out_height;


    def print_model_params(self):
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def save_model(self, sess, optimizer, epoch, suffix=""):
        postfix = '_epoch_{:04d}'.format(epoch)+ suffix;
        self.filepath = os.path.join(self.model_out_path, self.model_base_filename+ postfix + '.pth');
        print('self.filepath = ', self.filepath);
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            }, self.filepath);
        return self.filepath;

    def restore_model(self, sess=None, is_test=True, model_restore_filename=None):
        print('inside restore');
        if(not model_restore_filename is None):
            self.model_restore_filename = model_restore_filename
        if(self.model_restore_filename is None):
            self.filepath = None;
            return None;
        if(os.path.isfile(self.model_restore_filename)):
            self.filepath = self.model_restore_filename;
        elif(os.path.isfile(self.model_restore_filename + '.pth')):
            self.filepath = self.model_restore_filename + '.pth';
        else:
            self.filepath = os.path.join(self.model_out_path, self.model_restore_filename + '.pth');
        if(not os.path.isfile(self.filepath)):
            filepath_pattern = os.path.join(self.model_out_path, self.model_base_filename + '*.pth');
            list_of_files = glob.glob(filepath_pattern);
            if(len(list_of_files) <= 0):
                return None;
            self.filepath = max(list_of_files);
            print(self.filepath);
            if(not os.path.isfile(self.filepath)):
                return None;
        self.checkpoint = torch.load(self.filepath);
        self.checkpoint['model_state_dict']['od_stains']=self.state_dict()['od_stains'] ;
        self.load_state_dict(self.checkpoint['model_state_dict']);

        if(is_test):
            self.eval();
        else:
            self.train()
        
        return self.checkpoint;

    def change_mode(self, is_test=True):
        if(is_test):
            self.eval();
        else:
            self.train()

    def transform_intensity_to_optical_density(self, img_rgb):    
        od = -np.log((img_rgb+1)/255.0); 
        return od ;

    def init_layers(self, layer):    
        if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
            if(self.conv_init == 'normal'):
                torch.nn.init.normal_(layer.weight) ;
            elif(self.conv_init == 'xavier_uniform'):
                torch.nn.init.xavier_uniform_(layer.weight) ;
            elif(self.conv_init == 'xavier_normal'):
                torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
            elif(self.conv_init == 'he'):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
        elif(isinstance(layer, Iterable)):
            for sub_layer in layer:
                self.init_layers(sub_layer);