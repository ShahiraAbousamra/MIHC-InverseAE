import sys;
import os;
import glob;
from skimage import io; 
import numpy as np;
from skimage.transform import  resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import torch
_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 


text_2_rgb_id = {
    'cd16': [(255,255,255), 2, (0, 0, 1, 0, 0, 0, 0, 0), (30, 30, 30)], 
    'cd20': [(13,240,18), 5, (0, 0, 0, 0, 0, 1, 0, 0), (174, 38, 75)],
    'cd3': [(240,13,218), 4, (0, 0, 0, 0, 1, 0, 0, 0), (165, 168, 45)],
    'cd4': [(0,255,255), 3, (0, 0, 0, 1, 0, 0, 0, 0), (62, 147, 151)],
    'cd8': [(43,185,253), 1, (0, 1, 0, 0, 0, 0, 0, 0), (118, 62, 151)],
    'k17': [(227,137,26), 0, (1, 0, 0, 0, 0, 0, 0, 0), (151, 82, 62)],
    'hematoxilin': [(100, 80, 80), 6, (1, 0, 0, 0, 0, 0, 0, 0), (62, 104, 151)],
    'background': [(31,50,222), 7, (0, 0, 0, 0, 0, 0, 0, 1), (221, 220, 219)],
}

def transform_intensity_to_optical_density(img_rgb, const_val=255.0):
    img_rgb[np.where(img_rgb <5)] = 5;
    od = -np.log((img_rgb)/const_val);
    return od ;

lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True) # A low LPIPS score means that image patches are perceptual similar.

im_dir = '/mnt/data05/shared/sabousamra/mihc/datasets/Multiplex_dots_final_vis_im'
root = '/mnt/data05/shared/sabousamra/mihc/eval_96patches'
conc_dir_name = 'unsup_input_stain_wsi2_retrain_e3990_96patches'


#################### processing ##############################
in_dir = os.path.join(root, conc_dir_name)
out_dir = os.path.join(root, conc_dir_name + '_recon_quality')


if(not os.path.isdir(out_dir)):
    os.mkdir(out_dir)

# Create the stain matrix
stain_matrix_rgb = np.zeros((len(text_2_rgb_id), 3))
stain_matrix_od = np.zeros((len(text_2_rgb_id), 3)) # ns * 3
for key,val in text_2_rgb_id.items():
    stain_name = key
    stain_info = val
    stain_rgb = np.array(stain_info[3])
    stain_indx = stain_info[1]
    stain_matrix_rgb[stain_indx] = stain_rgb
    stain_matrix_od[stain_indx] = transform_intensity_to_optical_density(stain_rgb);

file_out_ssim = open(os.path.join(out_dir, 'ssim_matchstain'+'.txt'), 'w+')
file_out_psnr = open(os.path.join(out_dir, 'psnr_matchstain'+'.txt'), 'w+')
file_out_psnr_normalized = open(os.path.join(out_dir, 'psnr_normalized_matchstain'+'.txt'), 'w+')
file_out_mse = open(os.path.join(out_dir,  'mse_matchstain'+'.txt'), 'w+')
file_out_mse_normalized = open(os.path.join(out_dir, 'mse_normalized_matchstain'+'.txt'), 'w+')
file_out_lpips = open(os.path.join(out_dir,  'lpips_matchstain'+'.txt'), 'w+')


patch_files = glob.glob(os.path.join(im_dir, '*.png') );

for patch_file in patch_files:   
    base_filename  = os.path.splitext(os.path.basename(patch_file))[0];
    npy_filepath = glob.glob(os.path.join(in_dir, '*'+base_filename + '*.npy'))[0];
    print(patch_file)
    print(npy_filepath)
    target_img = io.imread(patch_file)
    if(not os.path.isfile(npy_filepath)):
        print('npy not found');
        continue;
    conc = np.load(npy_filepath, allow_pickle=True);# ns * h * w
    conc = conc.transpose((1,2,0)) # h * w * ns
    pred_img_od = np.matmul(conc.reshape(-1,conc.shape[-1]), stain_matrix_od).reshape((conc.shape[0], conc.shape[1], 3)) # (h*w, 3)
    pred_img_rgb = 255 * np.exp(-1 * pred_img_od)
    pred_img_rgb_resized = resize(pred_img_rgb.reshape((conc.shape[0], conc.shape[1], 3)), target_img.shape)
    print('pred_img_rgb_resized', pred_img_rgb_resized.shape)
    print('target_img', target_img.shape)
    io.imsave(os.path.join(out_dir, base_filename + '.png'), pred_img_rgb_resized.astype(np.uint8))


    # https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html
    print('pred_shape', torch.tensor(pred_img_rgb_resized.reshape(target_img.shape).transpose((2,0,1))/255).unsqueeze(0).shape)
    print('target shape', torch.tensor(target_img.transpose((2,0,1))/255).unsqueeze(0).shape)
    ssim_val = ssim(target_img, pred_img_rgb_resized, data_range=255, multichannel=True)
    psnr_val = psnr(target_img, pred_img_rgb_resized, data_range=255)
    mse_val = mse(target_img, pred_img_rgb_resized)
    psnr_normalized_val = psnr(target_img/255, pred_img_rgb_resized/255, data_range=1)
    mse_normalized_val = mse(target_img/255, pred_img_rgb_resized/255)
    lpips_val = lpips(torch.tensor(target_img.transpose((2,0,1))/255, dtype=torch.float).unsqueeze(0), torch.tensor(pred_img_rgb_resized.transpose((2,0,1))/255, dtype=torch.float).unsqueeze(0)).detach().cpu().numpy()
    print('ssim_val', ssim_val)
    print('psnr_val', psnr_val)
    print('mse_val', mse_val)
    print('psnr_normalized_val', psnr_normalized_val)
    print('mse_normalized_val', mse_normalized_val)
    print('lpips_val', lpips_val)
    file_out_ssim.write(base_filename + ',' + str(ssim_val) + '\n');
    file_out_ssim.flush();
    file_out_psnr.write(base_filename + ',' + str(psnr_val) + '\n');
    file_out_psnr.flush();
    file_out_psnr_normalized.write(base_filename + ',' + str(psnr_normalized_val) + '\n');
    file_out_psnr_normalized.flush();
    file_out_mse.write(base_filename + ',' + str(mse_val) + '\n');
    file_out_mse.flush();
    file_out_mse_normalized.write(base_filename + ',' + str(mse_normalized_val) + '\n');
    file_out_mse_normalized.flush();
    file_out_lpips.write(base_filename + ',' + str(lpips_val) + '\n');
    file_out_lpips.flush();


file_out_ssim.close();
file_out_psnr.close();
file_out_psnr_normalized.close();
file_out_mse.close();
file_out_mse_normalized.close();
file_out_lpips.close();



