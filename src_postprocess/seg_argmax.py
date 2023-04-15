import os
import numpy as np
from skimage import io;
import glob;
import cv2 ; 
import sys;
from skimage import filters


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





def argmax_all_conc_thresh_mask_hysteresis(conc_filepath, img_filepath, out_dir, resize_ratio, thresh_dict_high, thresh_dict_low, size_thresh_dict, do_thresh_size=False, do_fill_holes=False, do_dilate=False):
    img = None;
    if(os.path.isfile(img_filepath)):
        img = io.imread(img_filepath);
    out_filepath_argmax = os.path.join(out_dir, os.path.splitext(os.path.split(conc_filepath)[1])[0] + '_argmax' + '.npy');
    out_filepath_argmax_mz = os.path.join(out_dir, os.path.splitext(os.path.split(conc_filepath)[1])[0] + '_argmax_mz' + '.npy');
    conc_arr = np.load(conc_filepath, allow_pickle=True).astype(np.float32);

    conc_arr_new = []
    kernel=np.ones((3,3))

    # Resize to original size
    for i in range(conc_arr.shape[0]):
        if(not(img is None)):
            conc_arr_new.append(cv2.resize(conc_arr[i], (int(img.shape[1]*resize_ratio), int(img.shape[0]*resize_ratio)), interpolation = cv2.INTER_CUBIC));
        else:
            conc_arr_new.append(conc_arr[i]);

    conc_arr = np.stack(conc_arr_new, axis=0)
    conc_arr_argmax = conc_arr.argmax(axis=0);

    for stain_name, thresh_high in thresh_dict_high.items():
        stain_idx = text_2_rgb_id[stain_name][1];
        for s in range(conc_arr.shape[0]):
            if(s != stain_idx):
                b1 = conc_arr[stain_idx] >= thresh_high;
                b2 = conc_arr_argmax == stain_idx;
                conc_arr[s][np.where(np.logical_and(b1,b2))] = 0;

    for stain_name, thresh_high in thresh_dict_high.items():
        thresh_low = thresh_dict_low[stain_name];
        stain_idx = text_2_rgb_id[stain_name][1];
        binary_map = filters.apply_hysteresis_threshold(conc_arr[stain_idx], thresh_low, thresh_high)
        conc_arr[stain_idx] = binary_map * conc_arr[stain_idx]

    # none_arr will hold mask for areas that have none of the stains since we are now using the background as K17-neg
    # It is concatenated at the beginning of the conc array
    none_arr = (conc_arr.sum(axis=0)==0).astype(np.uint8);    
    none_arr = np.expand_dims(none_arr, axis=0);
    conc_arr = np.concatenate((none_arr, conc_arr), axis=0)
    print('conc_arr.shape = ', conc_arr.shape)

    conc_arr_argmax = conc_arr.argmax(axis=0);

    if(do_thresh_size or do_fill_holes):
        # Get rid of small detections and fill holes
        conc_arr_argmax_new = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
        for i in range(len(stain_names)):
            binary_mask = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
            binary_mask_out = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
            stain_idx = text_2_rgb_id[stain_names[i]][1] + 1;
            binary_mask[conc_arr_argmax == stain_idx]=255
            poly = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            try:
                contours,hia = poly
            except:
                im2, contours, hierarchy = poly;
            for idx in range(len(contours)):
                contour_i = contours[idx]
                physical_size = cv2.contourArea(contour_i)
                if do_thresh_size and physical_size<size_thresh_dict[stain_names[i]]:
                    continue
                cv2.drawContours(binary_mask_out, contours, idx, 255, -1)
            print('do_dilate',do_dilate)
            if(do_dilate):
                binary_mask_out = cv2.dilate(binary_mask_out, kernel, iterations=1)
            conc_arr_argmax_new[np.where(binary_mask_out == 255)] = stain_idx
        conc_arr_argmax = conc_arr_argmax_new;


    conc_arr_argmax.astype(np.uint8).dump(out_filepath_argmax);


    return;


def create_seg_map(filepath, out_dir, conc_thresh_dict, size_thresh_dict, resize_ratio, do_thresh_size=False, do_fill_holes=False, do_dilate=False):
    if(os.path.isfile(os.path.join(os.path.splitext(filepath)[0]+'.png'))):
        img = io.imread(os.path.join(os.path.splitext(filepath)[0]+'.png'));
    if(not os.path.isfile(os.path.join(os.path.splitext(filepath)[0]+'.npy'))):
        return;
    conc_arr = np.load(os.path.join(os.path.splitext(filepath)[0]+'.npy'), allow_pickle=True);    
    out_filepath_argmax = os.path.join(out_dir, os.path.splitext(os.path.split(filepath)[1])[0]  + '_argmax.npy');
    out_filepath_argmax_mz= os.path.join(out_dir, os.path.splitext(os.path.split(filepath)[1])[0]  + '_argmax_mz.npy');
    if(os.path.isfile(out_filepath_argmax)):
        return;
    out_filepath_conc = os.path.join(out_dir, os.path.splitext(os.path.split(filepath)[1])[0]  + '_conc.npy');
    out_filepath_img = os.path.join(out_dir, os.path.splitext(os.path.split(filepath)[1])[0]  + '_seg.png');

    conc_arr_new = []
    kernel=np.ones((3,3))

    # Resize to original size
    for i in range(conc_arr.shape[0]):
        if(resize_ratio != 1):
            conc_arr_new.append(cv2.resize(conc_arr[i], (int(img.shape[1]*resize_ratio), int(img.shape[0]*resize_ratio)), interpolation = cv2.INTER_CUBIC));
        else:
            conc_arr_new.append(conc_arr[i]);
    # Apply threshold
    for stain_name, thresh in conc_thresh_dict.items():
        stain_idx = text_2_rgb_id[stain_name][1];
        conc_arr_new[stain_idx][np.where(conc_arr_new[stain_idx] < thresh)] = 0;
    conc_arr_new = np.stack(conc_arr_new, axis=0)
    print('------------------------')
    print(conc_arr_new.shape)

    # none_arr will hold mask for areas that have none of the stains since we are now using the background as K17-neg
    # It is concatenated at the beginning of the conc array
    none_arr = (conc_arr_new.sum(axis=0)==0).astype(np.uint8);    
    none_arr = np.expand_dims(none_arr, axis=0);
    conc_arr_new = np.concatenate((none_arr, conc_arr_new), axis=0)
    print('conc_arr.shape = ', conc_arr_new.shape)

    # Get argmax
    conc_arr_argmax = conc_arr_new.argmax(axis=0);


    if(do_thresh_size or do_fill_holes):
        # Get rid of small detections and fill holes
        conc_arr_argmax_new = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
        for i in range(len(stain_names)):
            binary_mask = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
            binary_mask_out = np.zeros((conc_arr_argmax.shape[0],conc_arr_argmax.shape[1])).astype(np.uint8)
            stain_idx = text_2_rgb_id[stain_names[i]][1] + 1;
            binary_mask[conc_arr_argmax == stain_idx]=255
            poly = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            try:
                contours,hia = poly
            except:
                im2, contours, hierarchy = poly;
            for idx in range(len(contours)):
                contour_i = contours[idx]
                physical_size = cv2.contourArea(contour_i)
                if do_thresh_size and physical_size<size_thresh_dict[stain_names[i]]:
                    continue
                cv2.drawContours(binary_mask_out, contours, idx, 255, -1)
            print('do_dilate',do_dilate)
            if(do_dilate):
                binary_mask_out = cv2.dilate(binary_mask_out, kernel, iterations=1)
            conc_arr_argmax_new[np.where(binary_mask_out == 255)] = stain_idx
        conc_arr_argmax = conc_arr_argmax_new;

    conc_arr_argmax.astype(np.uint8).dump(out_filepath_argmax)


    return;




if __name__ == '__main__':

    stain_names = ['cd3', 'cd4', 'cd8', 'cd16', 'cd20', 'k17', 'hematoxilin'];

    size_thresh_dict = {'cd3':21, 'cd4':21, 'cd8':21, 'cd16':21, 'cd20':21, 'k17':21, 'hematoxilin':21, 'background':21}

    resize_ratio = 1


    ######## test 96 patches data #########
    root = '/mnt/data05/shared/sabousamra/mihc/eval_96patches'
    im_dir = '/mnt/data05/shared/sabousamra/mihc/datasets/Multiplex_dots_final_vis_im'

    conc_dir_name = 'unsup_input_stain_wsi2_retrain_e3990_96patches'
    thresh_dict_high = {'cd3':0.5, 'cd4':0.6, 'cd8':0.5, 'cd16':0.6, 'cd20':0.45, 'k17':0.5, 'hematoxilin':0.15, 'background':0.7} 
    thresh_dict_low = {'cd3':0.2, 'cd4':0.4, 'cd8':0.2, 'cd16':0.2, 'cd20':0.2, 'k17':0.1, 'hematoxilin':0.1, 'background':0.5}

    ######## Processing #########
    in_dir = os.path.join(root, conc_dir_name)
    out_dir = os.path.join(root, conc_dir_name + '_seg')


    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    im_files = glob.glob(os.path.join(im_dir,'*.png')) 
    for im_filepath in im_files:
        print('im_filepath',im_filepath)
        conc_filepath = glob.glob(os.path.join(in_dir, '*'+os.path.basename(im_filepath)[:-4]+'*.npy'))[0]
        argmax_all_conc_thresh_mask_hysteresis(conc_filepath, im_filepath, out_dir, resize_ratio, thresh_dict_high, thresh_dict_low, size_thresh_dict, do_thresh_size=True, do_fill_holes=True, do_dilate=True) 
        sys.stdout.flush()
