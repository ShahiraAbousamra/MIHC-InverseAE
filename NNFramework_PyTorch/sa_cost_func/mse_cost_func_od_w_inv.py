import sys;
import os;
import numpy as np;
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sa_helpers import multiplex_utils ;

class MSECostODWithInv:
    def __init__(self, kwargs):
        # Predefined list of arguments
        args = {'lambda_inv':1};

        args.update(kwargs);
        self.lambda_inv = float(args['lambda_inv']);


        self.cost_fn = nn.MSELoss();


    '''
    Calculate the loss as a weighted sum of the MSE(reconstructed image, input image) and MSE(reconstructed inverted image using inverted stains, inverted input image)
    Method Parameters:
        logits: the reconstructed images in the OD space.
        labels: the input images in the OD space.
        concentration_logits: the raw stain concentration maps output from the model.
        concentration_softmax: not used.
        stains_od: the reference stain vectors in OD space.
        deviceID: GPU device ID.
    Returns: the calculated loss.
    '''
    def calc_cost(self, logits, labels, concentration_logits, concentration_softmax, stains_od, deviceID):
        # Get input image in OD
        labels_od = multiplex_utils.transform_intensity_to_optical_density(labels)
        # Get MSE(reconstructed image, input image)
        loss0 = self.cost_fn(logits, labels_od) #*self.const_val;
        # Compute the inverse stains
        stains_rgb = multiplex_utils.transform_optical_density_to_intensity(stains_od);
        const_255 = torch.tensor([255.0]).to(deviceID);
        stains_rgb_inv = const_255 - stains_rgb ;
        stains_od_inv = multiplex_utils.transform_intensity_to_optical_density(stains_rgb_inv);
        # Flatten the raw concentration maps
        concentration_logits_flatten = concentration_logits.view(-1, stains_od.size()[1], concentration_logits.size()[2] * concentration_logits.size()[3])
        # Reconstructed inverted image using raw concentration maps and inverted stains
        logits_inv_flatten = torch.matmul(stains_od_inv, concentration_logits_flatten)
        logits_inv = logits_inv_flatten.view(-1, 3, concentration_logits.size()[2], concentration_logits.size()[3])
        labels_inv = 255 - labels;
        labels_inv_od = multiplex_utils.transform_intensity_to_optical_density(labels_inv)
        # Get MSE(reconstructed inverse image, inverted input image)
        loss1 = self.cost_fn(logits_inv, labels_inv_od) #*self.const_val;
        # Return weighted sum of losses
        return loss0 + self.lambda_inv * loss1;

    
