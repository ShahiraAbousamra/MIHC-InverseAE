from __future__ import print_function

from ..sa_net_data_provider import AbstractDataProvider;
from numpy import random;
from distutils.util import strtobool;

import numpy as np;
import glob;
import os;
from skimage import io;
from skimage import transform as sktransform;
import torchvision.transforms as transforms
import torchvision
import torch
import cv2 as cv;
import skimage.filters as skfilters;
import skimage.color as skcolor;

class MultiplexAutoencoderDataProviderRGB(AbstractDataProvider):
    def __init__(self, is_test, filepath_data, filepath_label, n_channels, n_classes, do_preprocess, do_augment, data_var_name=None, label_var_name=None, permute=False, repeat=True, kwargs={}):
        args = {'input_img_height':460, 'input_img_width': 700, 'file_name_suffix':''
            , 'pre_resize':'False', 'pre_center':'False', 'pre_edge':'False'
            , 'postprocess':'False', 'invert_img':'False', 'pad_y':0, 'pad_x':0};
        args.update(kwargs);    
        self.input_img_height = int(args['input_img_height']);
        self.input_img_width = int(args['input_img_width']);
        self.file_name_suffix = args['file_name_suffix'];
        self.pre_resize = bool(strtobool(args['pre_resize']));
        self.pre_center = bool(strtobool(args['pre_center']));
        self.pre_edge = bool(strtobool(args['pre_edge']));
        self.do_postprocess = bool(strtobool(args['postprocess']));
        self.invert_img = bool(strtobool(args['invert_img']));
        self.pad_y = int(args['pad_y']);
        self.pad_x = int(args['pad_x']);
        self.pad_y1 = int(np.floor(self.pad_y / 2.0));
        self.pad_y2 = int(np.ceil(self.pad_y / 2.0));
        self.pad_x1 = int(np.floor(self.pad_y / 2.0));
        self.pad_x2 = int(np.ceil(self.pad_y / 2.0));
        if(do_augment):
            self.create_augmentation_map(kwargs);
        if(self.do_postprocess):
            self.read_postprocess_parameters(kwargs); 

        self.is_test = is_test; # note that label will be None when is_test is true
        self.filepath_data = filepath_data;
        self.dir_data = os.path.split(self.filepath_data)[0] ;
        if(filepath_label == None or filepath_label.strip() == ''):
            self.filepath_label = None ;
        else:
            self.filepath_label = filepath_label ;
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.do_preprocess = do_preprocess;
        self.do_augment = do_augment;
        self.do_permute = permute;
        self.do_repeat = repeat;

        self.is_loaded = False;
        


        
    def load_data(self):
        self.data = None;
        self.label = None;
        self.last_fetched_indx = -1;
        self.permutation = None;
        self.data_count = 0;
        self.transform_data = transforms.Compose( 
            [transforms.ToTensor()]);

        img_path_files = glob.glob(os.path.join(self.filepath_data,'**', "*.png"), recursive=True);

        self.data_count = len(img_path_files);
        print('data_count')
        print(self.data_count)
        self.data = img_path_files;          


        # Permutation
        if(self.do_permute == True):
            self.permutation = np.random.permutation(self.data_count)
        else:
            self.permutation = None;

        self.is_loaded = True;


    def reset(self, repermute=None):
        self.last_fetched_indx = -1;
        if(repermute == True):
            self.do_permute = True;
            self.permutation = np.random.permutation(self.data_count);

    def get_next_one(self):
        ## Make sure data is loaded first
        if(self.is_loaded == False):
            self.load_data();
        
        ## Get next data point 
        self.last_fetched_indx = (self.last_fetched_indx + 1);
        if(self.do_repeat == False):
            if (self.last_fetched_indx >= self.data_count):
                return None;
        else:
            self.last_fetched_indx = self.last_fetched_indx % self.data_count;
        actual_indx = self.last_fetched_indx ;
        if(self.permutation is not None):
            actual_indx = self.permutation[self.last_fetched_indx];
        self.img_id = self.data[actual_indx]        
        

        data_point = self.load_image(self.img_id, do_match_size=(not self.do_preprocess and not self.do_postprocess));
 
        ## Process the data
        if(self.do_preprocess == True):
            data_point = self.preprocess(data_point);

        if(self.do_augment == True):
            data_point = self.augment(data_point);




        if(self.do_postprocess):
            data_point = self.postprocess(data_point);
        
        data_point = data_point.astype(np.float); ############################
        ##data_point /= 255;
        #data_point /= 4;
        #data_point -= 0.5;
        #data_point *= 2;
        ##print('np.transpose(data_point, (1, 2, 0)).shape = ', np.transpose(data_point, (2, 0, 1)).shape)
        #data_point = torch.tensor(np.transpose(data_point, (2,0, 1)));
        data_point = np.transpose(data_point, (2,0, 1)); 

       
        return data_point;

    ## Returns None, None if there is no more data to retrieve and repeat = false
    def get_next_n(self, n:int):
        ## Validate parameters
        if(n <= 0):
            return None, None;

        ## Make sure data is loaded first
        if(self.is_loaded == False):
            self.load_data();

        ## Get number of data points to retrieve        
        if(self.do_repeat == False):
            if (self.last_fetched_indx + n >= self.data_count):
                n = self.data_count - self.last_fetched_indx - 1;
                if(n <= 0):
                    return None, None;

        ## Get data shape
        data_size_x = self.input_img_width;
        data_size_y = self.input_img_height;    

        data_points = [];
    
        for i in range(0, n):
            d = self.get_next_one(); # returns the rgb image
            if(d is None):
                break;
            data_points.append(d);
        data_points = np.stack(data_points, axis=0)

        # Label is rgb - transform to whatever in train/loss
        labels = np.copy(data_points);
        #print('data_points.shape before = ', data_points.shape)
        # normalize the input
        #data_points /= 255;
        #data_points -= 0.5;
        #data_points *= 2;

        if(self.pad_y > 0 or self.pad_x > 0):
            data_points = np.pad(data_points, ((0,0),(0,0),(self.pad_y1, self.pad_y2),(self.pad_x1, self.pad_x2)),'constant', constant_values=128);

        data_points = torch.tensor(data_points, dtype = torch.float); # to avoid the error:  Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
        labels = torch.tensor(labels, dtype = torch.float)
        return data_points, labels;

    def preprocess(self, data_point):
        data_point2 = data_point;
        if(not(data_point.shape[0] == self.input_img_height) or not(data_point.shape[1] == self.input_img_width)):
            if(self.pre_resize):
                data_point2 = sktransform.resize(data_point, (self.input_img_height, self.input_img_width), preserve_range=True, anti_aliasing=True).astype(np.uint8);
            elif(self.pre_center):
                diff_y = self.input_img_height - data_point.shape[0];
                diff_x = self.input_img_width - data_point.shape[1];
                diff_y_div2 = diff_y//2;
                diff_x_div2 = diff_x//2;
                data_point2 = np.zeros((self.input_img_height, self.input_img_width, data_point.shape[2]));
                if(diff_y >= 0 and diff_x >= 0):
                    data_point2[diff_y:diff_y+self.input_img_height, diff_x:diff_x+self.input_img_width, :] = data_point;

            if(self.pre_edge):
                data_point2 = skcolor.rgb2gray(data_point2);
                data_point2 = skfilters.sobel(data_point2);
                data_point2 = data_point2.reshape(data_point2.shape[0], data_point2.shape[1], 1);
                data_point2 = np.concatenate((data_point2, data_point2, data_point2), axis=2);
                print(data_point2.shape);
        return data_point2;


    # Prepare the mapping from allowed operations to available operations index
    def create_augmentation_map(self, kwargs={}):
        args = {'aug_flip_h': 'True', 'aug_flip_v': 'True', 'aug_flip_hv': 'True' \
            , 'aug_rot180': 'True', 'aug_rot90': 'False', 'aug_rot270': 'False', 'aug_rot_rand': 'False' \
            , 'aug_brightness': 'False', 'aug_brightness_min': -50,  'aug_brightness_max': 50 \
            , 'aug_saturation': 'False', 'aug_saturation_min': -1.5,  'aug_saturation_max': 1.5 \
            , 'aug_hue': 'False', 'aug_hue_min': -50,  'aug_hue_max': 50 \
            , 'aug_scale': 'False', 'aug_scale_min': 1.0,  'aug_scale_max': 2.0 \
            , 'aug_translate': 'False',  'aug_translate_y_min': -20, 'aug_translate_y_max': 20,  'aug_translate_x_min': -20, 'aug_translate_x_max': 20
            };
        print(args);
        args.update(kwargs);    
        print(args);
        self.aug_flip_h = bool(strtobool(args['aug_flip_h']));
        self.aug_flip_v = bool(strtobool(args['aug_flip_v']));
        self.aug_flip_hv = bool(strtobool(args['aug_flip_hv']));
        self.aug_rot180 = bool(strtobool(args['aug_rot180']));
        self.aug_rot90 = bool(strtobool(args['aug_rot90']));
        self.aug_rot270 = bool(strtobool(args['aug_rot270']));
        self.aug_rot_random = bool(strtobool(args['aug_rot_rand']));
        self.aug_brightness = bool(strtobool(args['aug_brightness']));
        self.aug_saturation = bool(strtobool(args['aug_saturation']));
        self.aug_hue = bool(strtobool(args['aug_hue']));
        self.aug_scale = bool(strtobool(args['aug_scale']));
        self.aug_translate = bool(strtobool(args['aug_translate']));
        '''
        map allowed operation to the following values
        0: same (none)
        1: horizontal flip
        2: vertical flip
        3: horizontal and vertical flip
        4: rotate 180
        5: rotate 90
        6: rotate 270 or -90
        7: rotate random angle
        '''
        self.aug_map = {};
        self.aug_map[0] = 0; # (same) none 
        i = 1;
        if(self.aug_flip_h):
            self.aug_map[i] = 1;
            i += 1;
        if(self.aug_flip_v):
            self.aug_map[i] = 2;
            i += 1;
        if(self.aug_flip_hv):
            self.aug_map[i] = 3;
            i += 1;
        if(self.aug_rot180):
            self.aug_map[i] = 4;
            i += 1;
        if(self.aug_rot90):
            #print('self.aug_rot90={}'.format(self.aug_rot90));
            self.aug_map[i] = 5;
            i += 1;
        if(self.aug_rot270):
            #print('self.aug_rot270={}'.format(self.aug_rot270));
            self.aug_map[i] = 6;
            i += 1;
        if(self.aug_rot_random):
            #self.aug_map[i] = 7;
            self.aug_rot_min = int(args['aug_rot_min']);
            self.aug_rot_max = int(args['aug_rot_max']);
        if(self.aug_brightness):
        #    self.aug_map[i] = 7;
            self.aug_brightness_min = int(args['aug_brightness_min']);
            self.aug_brightness_max = int(args['aug_brightness_max']);
        #    i += 1;
        if(self.aug_saturation):
            self.aug_saturation_min = float(args['aug_saturation_min']);
            self.aug_saturation_max = float(args['aug_saturation_max']);
        if(self.aug_hue):
            self.aug_hue_min = int(args['aug_hue_min']);
            self.aug_hue_max = int(args['aug_hue_max']);
        if(self.aug_scale):
            self.aug_scale_min = float(args['aug_scale_min']);
            self.aug_scale_max = float(args['aug_scale_max']);
        if(self.aug_translate):
            self.aug_translate_y_min = int(args['aug_translate_y_min']);
            self.aug_translate_y_max = int(args['aug_translate_y_max']);
            self.aug_translate_x_min = int(args['aug_translate_x_min']);
            self.aug_translate_x_max = int(args['aug_translate_x_max']);
        print(self.aug_map)

    def augment(self, data_point):
        '''
            Select augmentation:        
            0: same (none)
            1: horizontal flip
            2: vertical flip
            3: horizontal and vertical flip
            4: rotate 180
            5: rotate 90
            6: rotate 270 or -90
            7: rotate random
        '''        
        # because width and height are not equal cannot do rotation 90 and 270
        #op = random.randint(0,7);
        #print('data_point.shape');
        #print(data_point.shape);
        #op = random.randint(0,5);

        # Select one of the valid operations and map it to its index in the available operations 
        op = random.randint(0,len(self.aug_map));
        op = self.aug_map[op];
        data_point2 = data_point;
        # Important: use ndarray.copy() when indexing with negative 
        #            It will alocate new memory for numpy array which make it normal, I mean the stride is not negative any more.
        #            Otherwise get the error: ValueError: some of the strides of a given numpy array are negative. This is currently not supported, but will be added in future releases.
        if(op == 1):
            data_point2 = data_point[:,::-1,:].copy();
        elif(op == 2):
            data_point2 = data_point[::-1,:,:].copy();
        elif(op == 3):
            data_point2 = data_point[:,::-1,:];
            data_point2 = data_point2[::-1,:,:].copy();
        elif(op == 4):
            data_point2 = np.rot90(data_point, k=2, axes=(0,1)).copy();
        elif(op == 5):
            data_point2 = np.rot90(data_point, k=1, axes=(0,1)).copy();
        elif(op == 6):
            data_point2 = np.rot90(data_point, k=3, axes=(0,1)).copy();
        if(self.aug_rot_random):
            angle = random.randint(self.aug_rot_min, self.aug_rot_max);
            data_point2 = sktransform.rotate(data_point2, angle, preserve_range=True).astype(np.uint8);


        op_brightness = random.random();
        op_saturation = random.random();
        op_hue = random.random();
        op_scale = random.random();
        if((self.aug_saturation and op_saturation > 0.5) or (self.aug_hue and op_hue > 0.5) or (self.aug_brightness and op_brightness > 0.5)):
            data_point2 = data_point2.astype(np.uint8);
            data_point2_hsv = cv.cvtColor(data_point2, cv.COLOR_RGB2HLS);
            data_point2_hsv = data_point2_hsv.astype(np.float)

            saturation = 1.0;
            hue = 0;
            brightness = 0;
            if(self.aug_hue and op_hue > 0.5):
                #hue = random.random()*(self.aug_hue_max-self.aug_hue_min) + self.aug_hue_min;
                hue = random.randint(self.aug_hue_min, self.aug_hue_max);
                data_point2_hsv[:,:,0] += hue;
                data_point2_hsv[:,:,0][np.where(data_point2_hsv[:,:,0] > 179)] = 179;
            if(self.aug_saturation and op_saturation > 0.5):
                #saturation = random.randint(self.aug_saturation_min, self.aug_saturation_max);
                saturation = random.random() * (self.aug_saturation_max-self.aug_saturation_min) + self.aug_saturation_min;
                data_point2_hsv[:,:,2] *= saturation;
                data_point2_hsv[:,:,2][np.where(data_point2_hsv[:,:,2] > 255)] = 255;
            if(self.aug_brightness and op_brightness > 0.5):
                brightness = random.randint(self.aug_brightness_min, self.aug_brightness_max);
                data_point2_hsv[:,:,1] += brightness;
                data_point2_hsv[:,:,1][np.where(data_point2_hsv[:,:,1] > 255)] = 255;

            # The ranges that OpenCV manage for HSV format are the following:
            # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. Different softwares use different scales.
            data_point2_hsv[np.where(data_point2_hsv < 0)] = 0;
            data_point2_hsv = data_point2_hsv.astype(np.uint8);
            data_point2 = cv.cvtColor(data_point2_hsv, cv.COLOR_HLS2RGB);
        if(self.aug_translate):
            translate_y = np.random.randint(self.aug_translate_y_min, high = self.aug_translate_y_max);
            translate_x = np.random.randint(self.aug_translate_x_min, high = self.aug_translate_x_max);
            translate_transform = sktransform.AffineTransform(translation = (translate_x, translate_y));      
            data_point2 = sktransform.warp(data_point2, translate_transform, preserve_range=True).astype(np.uint8); 
        if(self.aug_scale and op_scale > 0.5):
            scale = random.random()*(self.aug_scale_max-self.aug_scale_min) + self.aug_scale_min;
            data_point2 = sktransform.rescale(data_point2, scale, preserve_range=True).astype(np.uint8);        
            scale_height, scale_width,_ = data_point2.shape;
            diff_height = scale_height - self.input_img_height;
            diff_width = scale_width - self.input_img_width;
            start_y = 0;
            start_x = 0;
            if(diff_height > 0):
                start_y = random.randint(0, diff_height);
            if(diff_width > 0):
                start_x = random.randint(0, diff_width);
            data_point2 = data_point2[start_y : start_y+self.input_img_height, start_x : start_x+self.input_img_width, : ]

        return data_point2;

    def load_image(self, filepath, do_match_size=False):
        img = io.imread(filepath);
        if(img.shape[2] > 3): # Remove the alpha channel
            img = img[:,:,0:3];
        if(do_match_size):
            if((not (img.shape[0] == self.input_img_height)) \
                or (not (img.shape[1] == self.input_img_width))):

                valid_height = self.input_img_height;
                valid_width = self.input_img_width;
                if(img.shape[0] < self.input_img_height):
                    valid_height = img.shape[0];
                if(img.shape[1] < self.input_img_width):
                    valid_width = img.shape[1];

                img_new = np.zeros((self.input_img_height, self.input_img_width, img.shape[2]));
                img_new[0:valid_height, 0:valid_width, :] = img[0:valid_height, 0:valid_width, :];
                img = img_new;
        if(self.invert_img):
            img = 255 - img;
        img[np.where(img <5)] = 5;
        img[np.where(img >250)] = 250;

        return img;


    def pytorch_to_tf_format(self, img):
        # Channels, height, width --> height, width, channels 
        return np.transpose(img, (1, 2, 0));

    def tf_to_pytorch_format(self, img):
        # Height, width, channels --> channels, height, width
        return np.transpose(img, (2, 0, 1));

    def read_postprocess_parameters(self, kwargs={}):
        args = {'post_resize': 'False', 'post_crop_center': 'False',
            'post_crop_height': 128, 'post_crop_width': 128
            };
        print(args);
        args.update(kwargs);    
        print(args);
        self.post_resize = bool(strtobool(args['post_resize']));
        self.post_crop_center = bool(strtobool(args['post_crop_center']));
        if(self.post_crop_center ):
            self.post_crop_height = int(args['post_crop_height']);
            self.post_crop_width = int(args['post_crop_width']);

    def postprocess(self, data_point):
        data_point2 = data_point.copy();

        if(self.post_crop_center):
            starty = (data_point2.shape[0] - self.post_crop_height)//2;
            startx = (data_point2.shape[1] - self.post_crop_width)//2;
            endy = starty + self.post_crop_height;
            endx = startx + self.post_crop_width;
            if(starty < 0 or startx < 0): # In case rotated the width and height will have changed
                starty = (data_point2.shape[0] - self.post_crop_width)//2;
                startx = (data_point2.shape[1] - self.post_crop_height)//2;
                endy = starty + self.post_crop_width;
                endx = startx + self.post_crop_height;
            
            data_point2 = data_point2[starty:endy, startx:endx, :];

        if(self.post_resize):
            data_point2 = sktransform.resize(data_point2, (self.input_img_height, self.input_img_width), preserve_range=True, anti_aliasing=True);

        return data_point2;


