import numpy as np
import sys
sys.path.append("..");
sys.path.append(".");
sys.path.append("../..");
sys.path.append("...");
from NNFramework_PyTorch.sa_runners import multiplex_autoencoder_runner_pytorch;

'''
cd /mnt/data05/shared/sabousamra/mihc/unsup_public_src/NNFramework_Pytorch_external_call

Run a command of the form:
CUDA_VISIBLE_DEVICES='1' nohup  python ./external_run.py <config file path> 0  >> <output log file path>&
Example:
CUDA_VISIBLE_DEVICES='1' nohup  python ./external_run.py ../NNFramework_PyTorch/config_multiplex_ae1.0_train/config_multiplex-ae_rgb2stains_simple-w-mp3_nocrop_inv_aux_task_input_stain_wsi2_split1.ini 0  >> /mnt/data05/shared/sabousamra/mihc/checkpoints/rgb2stains_inv_aux_task_input_stain_wsi2_split1/log.txt&

'''

if __name__ == "__main__":
    
    multiplex_autoencoder_runner_pytorch.main(sys.argv[0:])