import os
import numpy as np
from skimage import io;
import glob;
import cv2 ; 
import sys;

text_2_rgb_id = {
    'cd16': [(255,255,255), 2, (0, 0, 1, 0, 0, 0, 0, 0), (30, 30, 30)],
    'cd20': [(13,240,18), 5, (0, 0, 0, 0, 0, 1, 0, 0), (174, 38, 75)],
    'cd3': [(240,13,218), 4, (0, 0, 0, 0, 1, 0, 0, 0), (165, 168, 45)],
    'cd4': [(0,255,255), 3, (0, 0, 0, 1, 0, 0, 0, 0), (62, 147, 151)],
    'cd8': [(43,185,253), 1, (0, 1, 0, 0, 0, 0, 0, 0), (118, 62, 151)],
    'k17': [(227,137,26), 0, (1, 0, 0, 0, 0, 0, 0, 0), (151, 82, 62)],
    'hematoxilin': [(100, 80, 80), 6, (1, 0, 0, 0, 0, 0, 0, 0), (62, 104, 151)],
    'background': [(31,50,222), 7, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210)],
}

def transform_intensity_to_optical_density(img_rgb, const_val=255.0):  
    if(not isinstance(img_rgb , np.ndarray)):
        img_rgb = np.array(img_rgb);
    img_rgb[np.where(img_rgb <5)] = 5;
    od = -np.log((img_rgb)/const_val); 
    return od ;

def transform_optical_density_to_intensity(od, const_val=255.0):    
    if(not isinstance(od , np.ndarray)):
        od = np.array(od);
    rgb = np.exp(-od)*const_val 
    return rgb ;

def visualize_binary_mask(imgfilepath, argmax_filepath, out_dir):
    out_basename = os.path.splitext(os.path.basename(argmax_filepath))[0]
    stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin']
    argmax_arr = np.load(argmax_filepath, allow_pickle=True);
    img = io.imread(imgfilepath);

    argmax_arr = cv2.resize(argmax_arr, (int(img.shape[1]),int(img.shape[0])), interpolation = cv2.INTER_NEAREST)
    argmax_arr.astype(np.uint8).dump(os.path.join(out_dir, out_basename+'.npy'))

    for i in range(len(stain_names)):
        stain_seg_binary = np.zeros(argmax_arr.shape)
        stain_name = stain_names[i];
        stain_idx = text_2_rgb_id[stain_name][1] + 1;
        stain_seg_binary[np.where(argmax_arr== stain_idx)]  = 1
        stain_seg_binary.astype(np.uint8).dump(os.path.join(out_dir, out_basename + '_' + stain_name + '_binary.npy'))
        io.imsave(os.path.join(out_dir, out_basename + '_' + stain_name + '_binary.png'), (stain_seg_binary*255).astype(np.uint8))


def visualize_raw_stain_conc_map(conc_filepath, out_dir):
    stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin']
    conc_arr = np.load(conc_filepath, allow_pickle=True);
    for i in range(len(stain_names)):
        stain_name = stain_names[i];
        out_filepath = os.path.join(out_dir, os.path.splitext(os.path.split(conc_filepath)[1])[0] + '_'+stain_name + '_rawmap.png');
        stain_rgb = text_2_rgb_id[stain_name][3];
        stain_od = transform_intensity_to_optical_density(stain_rgb);
        stain_idx = text_2_rgb_id[stain_name][1];
        stain_conc_arr = conc_arr[stain_idx];
        stain_conc_img_od = np.matmul(stain_od.reshape((-1,1)), stain_conc_arr.reshape(1,-1));
        stain_conc_img_od = np.transpose(stain_conc_img_od, axes=(1,0))
        stain_conc_img_od = stain_conc_img_od.reshape((conc_arr.shape[1],conc_arr.shape[2],3))
        stain_conc_img_rgb = transform_optical_density_to_intensity(stain_conc_img_od).astype(np.uint8);
        io.imsave(out_filepath, stain_conc_img_rgb);


def visualize_mask_stain_conc_map(conc_filepath, argmax_filepath, imgfilepath, out_dir, bg_color=(255,255,255)):
    stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin']
    conc_arr = np.load(conc_filepath, allow_pickle=True);
    argmax_arr = np.load(argmax_filepath, allow_pickle=True);
    img = io.imread(imgfilepath);
    argmax_arr = cv2.resize(argmax_arr, (int(img.shape[1]),int(img.shape[0])), interpolation = cv2.INTER_NEAREST)
    for i in range(len(stain_names)):
        stain_name = stain_names[i];
        out_filepath_bg_black = os.path.join(out_dir, os.path.splitext(os.path.split(conc_filepath)[1])[0] + '_'+stain_name + '_concmask_bg_black.png');
        out_filepath_bg_white = os.path.join(out_dir, os.path.splitext(os.path.split(conc_filepath)[1])[0] + '_'+stain_name + '_concmask_bg_white.png');
        stain_rgb = text_2_rgb_id[stain_name][3];
        stain_od = transform_intensity_to_optical_density(stain_rgb);

        # Get raw concentration map
        stain_idx = text_2_rgb_id[stain_name][1];
        stain_conc_arr = conc_arr[stain_idx];
        stain_conc_arr = cv2.resize(stain_conc_arr, (int(img.shape[1]),int(img.shape[0])), interpolation = cv2.INTER_NEAREST)
        # Multiply
        stain_conc_img_od = np.matmul(stain_od.reshape((-1,1)), stain_conc_arr.reshape(1,-1));
        stain_conc_img_od = np.transpose(stain_conc_img_od, axes=(1,0))
        stain_conc_img_od = stain_conc_img_od.reshape((int(img.shape[0]),int(img.shape[1]),3))
        stain_conc_img_rgb = transform_optical_density_to_intensity(stain_conc_img_od).astype(np.uint8);

        # Get binary mask
        stain_seg_binary = np.zeros(argmax_arr.shape)
        stain_idx = text_2_rgb_id[stain_name][1] + 1;
        stain_seg_binary[np.where(argmax_arr== stain_idx)]  = 1
        stain_seg_binary = stain_seg_binary[:,:,np.newaxis]

        stain_conc_mask_bg_black = stain_conc_img_rgb.copy()
        stain_conc_mask_bg_white = stain_conc_img_rgb.copy()
        stain_conc_mask_bg_black[np.where(argmax_arr!= stain_idx)] = [0,0,0]
        stain_conc_mask_bg_white[np.where(argmax_arr!= stain_idx)] = [255,255,255]

        # Save
        io.imsave(out_filepath_bg_black, stain_conc_mask_bg_black.astype(np.uint8));
        io.imsave(out_filepath_bg_white, stain_conc_mask_bg_white.astype(np.uint8));


def visualize_mask_img(imgfilepath, argmax_filepath, out_dir):
    out_basename = os.path.splitext(os.path.basename(argmax_filepath))[0]
    stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin']
    argmax_arr = np.load(argmax_filepath, allow_pickle=True);
    img = io.imread(imgfilepath);

    argmax_arr = cv2.resize(argmax_arr, (int(img.shape[1]),int(img.shape[0])), interpolation = cv2.INTER_NEAREST)

    for i in range(len(stain_names)):
        stain_name = stain_names[i];
        stain_idx = text_2_rgb_id[stain_name][1] + 1;
        out_filepath_bg_black = os.path.join(out_dir, out_basename + '_'+stain_name + '_imgmask_bg_black.png');
        out_filepath_bg_white = os.path.join(out_dir, out_basename + '_'+stain_name + '_imgmask_bg_white.png');

        img_bg_black = img.copy();
        img_bg_white = img.copy();
        img_bg_black[np.where(argmax_arr!= stain_idx)] = [0,0,0]
        img_bg_white[np.where(argmax_arr!= stain_idx)] = [255,255,255]

        io.imsave(out_filepath_bg_black, img_bg_black.astype(np.uint8))
        io.imsave(out_filepath_bg_white, img_bg_white.astype(np.uint8))


def visualize_all_stains_in_single_mask(imgfilepath, argmax_filepath, out_dir):
    argmax_arr = np.load(argmax_filepath, allow_pickle=True);
    img = io.imread(imgfilepath);
    argmax_arr = cv2.resize(argmax_arr, (int(img.shape[1]),int(img.shape[0])), interpolation = cv2.INTER_NEAREST)
    basefilename = os.path.splitext(os.path.basename(argmax_filepath))[0];    
    img_mask = np.zeros(img.shape)
    img_maskh = np.zeros(img.shape)
    for stain_name in text_2_rgb_id.keys():
        stain_indx = text_2_rgb_id[stain_name][1]+1
        stain_rgb = text_2_rgb_id[stain_name][3]
        img_maskh[np.where(argmax_arr==stain_indx)] = stain_rgb
        if(stain_name == 'hematoxilin'):
            stain_rgb=(212, 212, 210)
        img_mask[np.where(argmax_arr==stain_indx)] = stain_rgb
    io.imsave(os.path.join(out_dir, basefilename +'_allstains_onemask.png'), img_mask.astype(np.uint8));
    io.imsave(os.path.join(out_dir, basefilename +'_allstains_onemaskh.png'), img_maskh.astype(np.uint8));

def visualize_raw_stain_conc_map_by_threshold(conc_filepath, out_dir, threshold_arr):
    stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin', 'background']
    conc_arr = np.load(conc_filepath, allow_pickle=True);
    for i in range(len(stain_names)):
        stain_name = stain_names[i];
        stain_rgb = text_2_rgb_id[stain_name][3];
        stain_od = transform_intensity_to_optical_density(stain_rgb);
        stain_idx = text_2_rgb_id[stain_name][1];
        stain_conc_arr = conc_arr[stain_idx];
        stain_conc_img_od = np.matmul(stain_od.reshape((-1,1)), stain_conc_arr.reshape(1,-1));
        stain_conc_img_od = np.transpose(stain_conc_img_od, axes=(1,0))
        stain_conc_img_od = stain_conc_img_od.reshape((conc_arr.shape[1],conc_arr.shape[2],3))
        stain_conc_img_rgb = transform_optical_density_to_intensity(stain_conc_img_od).astype(np.uint8);
        for th in threshold_arr:
            out_filepath = os.path.join(out_dir, os.path.splitext(os.path.split(conc_filepath)[1])[0] + '_'+stain_name + '_rawmap'+'_th'+str(th)+'.png');
            binary_map = (stain_conc_arr >= th)
            binary_map = binary_map[:,:,np.newaxis]
            io.imsave(out_filepath, (stain_conc_img_rgb*binary_map).astype(np.uint8));




if __name__=="__main__":

    ######## test data config #########
    root = '/mnt/data05/shared/sabousamra/mihc/eval_96patches'
    im_dir = '/gpfs/projects/KurcGroup/sabousamra/multiplex1.0/datasets/Multiplex_dots_final_vis_im'
    conc_dir_name = 'unsup_e5970_96patches'


    ######## Processing #########
    conc_dir = os.path.join(root, conc_dir_name)
    out_dir_raw = os.path.join(root, conc_dir_name + '_vis')
    out_dir_th = os.path.join(root, conc_dir_name + '_thresholds_vis')

    threshold_arr = np.array([0.00, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])

    if(not os.path.isdir(out_dir_raw)):
        os.mkdir(out_dir_raw)

    if(not os.path.isdir(out_dir_th)):
        os.mkdir(out_dir_th)

    im_files = glob.glob(os.path.join(im_dir,'*.png')) 
    for im_filepath in im_files:
        print('im_filepath',im_filepath)
        conc_filepath = glob.glob(os.path.join(conc_dir, '*'+os.path.basename(im_filepath)[:-4]+'*.npy'))[0]
        print('conc_filepath',conc_filepath)
        sys.stdout.flush()
        visualize_raw_stain_conc_map(conc_filepath, out_dir_raw)
        visualize_raw_stain_conc_map_by_threshold(conc_filepath, out_dir_th, threshold_arr)
        sys.stdout.flush()


