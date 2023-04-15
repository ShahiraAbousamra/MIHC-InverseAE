import numpy as np;
from enum import Enum;
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNArchUtilsPyTorch:
    
    @staticmethod
    def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, input_width=-1, input_height=-1):
        conv_2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias);
        output_width = -1;
        output_height = -1;
        if(input_width > 0):
            output_width = input_width - kernel_size + 1;
        if(input_height > 0):
            output_height = input_height - kernel_size + 1;
        print('conv_2d shape')
        print(conv_2d.shape)
        return conv_2d, output_width, output_height;


    @staticmethod
    def crop_a_to_b(input_a, input_b):
        shape_a = input_a.size();
        shape_b = input_b.size();
        cropped = input_a[:, :, (shape_a[2]-shape_b[2])//2 : (shape_a[2]-shape_b[2])//2 + shape_b[2], (shape_a[3]-shape_b[3])//2 : (shape_a[3]-shape_b[3])//2 + shape_b[3]]

        return cropped;

