import os
import numpy as np
from skimage import io;
import glob;
import cv2 ; 
import sys;
from skimage.measure import label, moments

text_2_rgb_id = {
    'cd16': [(255,255,255), 2, (0, 0, 1, 0, 0, 0, 0, 0), (30, 30, 30), (0, 0, 0)],
    'cd20': [(13,240,18), 5, (0, 0, 0, 0, 0, 1, 0, 0), (174, 38, 75), (255, 0, 0)],
    'cd3': [(240,13,218), 4, (0, 0, 0, 0, 1, 0, 0, 0), (165, 168, 45), (255, 255, 0)],
    'cd4': [(0,255,255), 3, (0, 0, 0, 1, 0, 0, 0, 0), (62, 147, 151), (0, 255, 255)],
    'cd8': [(43,185,253), 1, (0, 1, 0, 0, 0, 0, 0, 0), (118, 62, 151), (160, 40, 200)],
    'k17': [(227,137,26), 0, (1, 0, 0, 0, 0, 0, 0, 0), (151, 82, 62), (128, 64, 0)],
    'hematoxilin': [(100, 80, 80), 6, (1, 0, 0, 0, 0, 0, 0, 0), (62, 104, 151), (62, 104, 151)],
    'background': [(31,50,222), 7, (0, 0, 0, 0, 0, 0, 0, 1), (212, 212, 210), (212, 212, 210)],
}



def compute_fscore_stats_from_gt_dots(img_filepath, argmax_arr, gt_dots_arr, out_dir, stain_names_lst, file_log, do_visualize=True):
    img_arr = io.imread(img_filepath)
    img_name = os.path.splitext(os.path.basename(img_filepath))[0]
    argmax_arr = cv2.resize(argmax_arr, (int(img_arr.shape[1]),int(img_arr.shape[0])), interpolation = cv2.INTER_NEAREST)

    img_gt_all_centers = img_arr.copy()
    img_et_all_centers = img_arr.copy()

    tp_arr = np.zeros((len(stain_names_lst)), dtype=np.int)
    fp_arr = np.zeros((len(stain_names_lst)), dtype=np.int)
    fn_arr = np.zeros((len(stain_names_lst)), dtype=np.int)

    for s in range(len(stain_names_lst)):
        # Get stain info
        stain_name = stain_names_lst[s] 
        stain_info = text_2_rgb_id[stain_name]
        complement_clr, stain_indx, hot_encode, stain_rgb, stain_display_rgb = stain_info

        # Initialize arrays
        gt_dots = (gt_dots_arr == stain_indx)
        et_map = (argmax_arr == stain_indx+1)
        et_dots = np.zeros(argmax_arr.shape)
        et_comp_mask = label(et_map)
        img_et_stain = img_arr.copy()
    

        # Find estimated components centers and generate dot map and overlay on image
        contours, hierarchy = cv2.findContours((et_map*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for idx in range(len(contours)):
            contour_i = contours[idx]
            M = cv2.moments(contour_i)
            if(M['m00'] == 0):
                continue;
            cx = round(M['m10'] / M['m00'])
            cy = round(M['m01'] / M['m00'])
            et_dots[cy, cx] = 1
            img_et_all_centers[cy-4:cy+5, cx-4:cx+5,:] = (255,255,255)
            img_et_all_centers[cy-2:cy+3, cx-2:cx+3,:] = stain_display_rgb 

        # Find gt components centers and overlay on image
        gt_centers = np.where(gt_dots > 0)
        for idx in range(len(gt_centers[0])):
            cx = gt_centers[1][idx]
            cy = gt_centers[0][idx]
            img_gt_all_centers[cy-4:cy+5, cx-4:cx+5,:] = (255,255,255)
            img_gt_all_centers[cy-2:cy+3, cx-2:cx+3,:] = stain_display_rgb 

        # g_dots_remain will be used to find remaining dots during matching with estimation and stats calculation
        g_dots_remain = gt_dots.copy()

        for l in range(1, et_comp_mask.max()+1):
            # Get component l and its center
            et_comp_mask_l = (et_comp_mask == l)
            M = moments(et_comp_mask_l)
            (y,x) = int(M[1, 0] / M[0, 0]), int(M[0, 1] / M[0, 0])
            # Find gt matches with current component (there can be more than one match because we do not have single instance detection)
            tp = (et_comp_mask_l * g_dots_remain).sum()
            if (tp > 0): # true pos
                tp_arr[s] += tp
                (yg,xg) = np.where((et_comp_mask_l * g_dots_remain) > 0)
                for c in range(len(yg)):
                    yg_c = yg[c]
                    xg_c = xg[c]
                    g_dots_remain[yg_c, xg_c] = 0 
                    img_et_stain[max(0,yg_c-4):min(yg_c+5, g_dots_remain.shape[0]), max(0,xg_c-4):min(xg_c+5, g_dots_remain.shape[1]),:] = [255,255,255]
                    img_et_stain[max(0,yg_c-2):min(yg_c+3, g_dots_remain.shape[0]), max(0,xg_c-2):min(xg_c+3, g_dots_remain.shape[1]), :] = [0,255,0]
            else: # false pos
                fp_arr[s] += 1
                img_et_stain[max(0,y-4):min(y+5, g_dots_remain.shape[0]), max(0,x-4):min(x+5, g_dots_remain.shape[1]),:] = [255,255,255]
                img_et_stain[max(0,y-2):min(y+3, g_dots_remain.shape[0]), max(0,x-2):min(x+3, g_dots_remain.shape[1]), :] = [0,0,255]

        fn_points = np.where(g_dots_remain > 0)
        fn_count = len(fn_points[0])
        fn_arr[s] += fn_count
        for p in range(len(fn_points[0])):  # false neg
            y = fn_points[0][p]
            x = fn_points[1][p]                    
            img_et_stain[max(0,y-4):min(y+5, g_dots_remain.shape[0]), max(0,x-4):min(x+5, g_dots_remain.shape[1]),:] = [255,255,255]
            img_et_stain[max(0,y-2):min(y+3, g_dots_remain.shape[0]), max(0,x-2):min(x+3, g_dots_remain.shape[1]), :] = [255,0,0]

        if(do_visualize):
            io.imsave(os.path.join(out_dir, img_name + '_et_'+stain_name+'_f.png'), img_et_stain)

    if(do_visualize):
        io.imsave(os.path.join(out_dir, img_name + '_gt_all_centers.png'), img_gt_all_centers)
        io.imsave(os.path.join(out_dir, img_name + '_et_all_centers.png'), img_et_all_centers)

    return tp_arr, fp_arr, fn_arr

if __name__=="__main__":
    stain_names_lst = ['k17', 'cd8', 'cd16', 'cd4', 'cd3', 'cd20'];

    root =   '/mnt/data05/shared/sabousamra/mihc/eval_96patches'
    im_dir = '/mnt/data05/shared/sabousamra/mihc/datasets/Multiplex_dots_final_vis_im'
    gt_dir = '/mnt/data05/shared/sabousamra/mihc/datasets/Multiplex_dots_final_vis'

    argmax_dir = os.path.join(root,'unsup_input_stain_wsi2_retrain_e3990_96patches_seg')
    out_dir_fscore = os.path.join(root,'unsup_input_stain_wsi2_retrain_e3990_96patches_fscore')
    do_visualize=True


    ################# Processing ####################

    if(not os.path.isdir(out_dir_fscore)):
        os.mkdir(out_dir_fscore)

    log_filepath = os.path.join(out_dir_fscore, 'out.txt')
    i = 1
    while(os.path.exists(log_filepath)):
        log_filepath = os.path.join(out_dir_fscore, 'out'+str(i)+'.txt')
        i += 1
    with open(log_filepath, 'w') as file_log:
    
        tp_arr = np.zeros((len(stain_names_lst)), dtype=np.int)
        fp_arr = np.zeros((len(stain_names_lst)), dtype=np.int)
        fn_arr = np.zeros((len(stain_names_lst)), dtype=np.int)

        print('stain_names_lst',stain_names_lst)
        file_log.write('stain_names_lst ' + str(stain_names_lst)+ '\n')
        print('im_dir',im_dir)
        file_log.write('im_dir ' + str(im_dir)+ '\n')
        print('gt_dir',gt_dir)
        file_log.write('gt_dir ' + str(gt_dir)+ '\n')
        print('argmax_dir',argmax_dir)
        file_log.write('argmax_dir ' + str(argmax_dir)+ '\n')
        print('out_dir_fscore',out_dir_fscore)
        file_log.write('out_dir_fscore ' + str(out_dir_fscore)+ '\n')
        np.array(stain_names_lst).dump(os.path.join(out_dir_fscore,'stain_names_list.npy'))

        im_files = glob.glob(os.path.join(im_dir,'*.png')) 
        for im_filepath in im_files:
            print('im_filepath',im_filepath)
            file_log.write('im_filepath ' + str(im_filepath)+ '\n')
            sys.stdout.flush()
            file_log.flush()
            image_name = os.path.splitext(os.path.basename(im_filepath))[0]
            gt_filepath = glob.glob(os.path.join(gt_dir, '*'+os.path.basename(im_filepath)[:-4]+'*.npy'))[0]
            print('argmax_filepath pattern', os.path.join(argmax_dir, '*'+os.path.basename(im_filepath)[:-4]+'*.npy'))
            argmax_filepath = glob.glob(os.path.join(argmax_dir, '*'+os.path.basename(im_filepath)[:-4]+'*.npy'))[0]
            gt_arr = np.load(gt_filepath, allow_pickle=True)[:,:,-1]
            argmax_arr = np.load(argmax_filepath, allow_pickle=True)
        
            tp_arr_im, fp_arr_im, fn_arr_im = compute_fscore_stats_from_gt_dots(im_filepath, argmax_arr, gt_arr, out_dir_fscore, stain_names_lst, file_log, do_visualize=do_visualize)
            sys.stdout.flush()
            file_log.flush()

            tp_arr = tp_arr + tp_arr_im
            fp_arr = fp_arr + fp_arr_im
            fn_arr = fn_arr + fn_arr_im
            print(image_name, 'tp', tp_arr_im )
            file_log.write(im_filepath  + ' tp ' + str(tp_arr_im)+ '\n')
            print(image_name, 'fp', fp_arr_im )
            file_log.write(im_filepath  + ' fp ' + str(fp_arr_im)+ '\n')
            print(image_name, 'fn', fn_arr_im )
            file_log.write(im_filepath  + ' fn ' + str(fn_arr_im)+ '\n')
            tp_arr_im.dump(os.path.join(out_dir_fscore, image_name + '_tp.npy'))
            fp_arr_im.dump(os.path.join(out_dir_fscore, image_name + '_fp.npy'))
            fn_arr_im.dump(os.path.join(out_dir_fscore, image_name + '_fn.npy'))
            # compute image fscore
            for s in range(len(stain_names_lst)):
                stain_name = stain_names_lst[s]
                tp = tp_arr_im[s]
                fp = fp_arr_im[s]
                fn = fn_arr_im[s]
                if(tp+fp == 0):
                    precision = 1
                else:
                    precision = tp/(tp+fp)
                if(tp+fn == 0):
                    recall = 1
                else:
                    recall = tp/(tp+fn)
                if(precision+recall == 0):
                    fscore = 1
                else:
                    fscore = 2*precision*recall/(precision+recall)
                print('img', image_name, stain_name, 'precision', precision, 'recall', recall, 'fscore', fscore )
                file_log.write('img ' + str(image_name) + ' ' + str(stain_name) + ' precision ' + str(precision) + ' recall ' + str(recall) + ' fscore ' + str(fscore) + '\n')
                sys.stdout.flush()
                file_log.flush()

        tp_arr.dump(os.path.join(out_dir_fscore, 'all' + '_tp.npy'))
        fp_arr.dump(os.path.join(out_dir_fscore, 'all' + '_fp.npy'))
        fn_arr.dump(os.path.join(out_dir_fscore, 'all' + '_fn.npy'))
        # Compute overall fscore
        for s in range(len(stain_names_lst)):
            stain_name = stain_names_lst[s]
            tp = tp_arr[s]
            fp = fp_arr[s]
            fn = fn_arr[s]
            if(tp+fp == 0):
                precision = 1
            else:
                precision = tp/(tp+fp)
            if(tp+fn == 0):
                recall = 1
            else:
                recall = tp/(tp+fn)
            if(precision+recall == 0):
                fscore = 1
            else:
                fscore = 2*precision*recall/(precision+recall)
            print('overall', stain_name, 'precision', precision, 'recall', recall, 'fscore', fscore )
            file_log.write('overall ' + str(stain_name) + ' precision ' + str(precision) + ' recall ' + str(recall) + ' fscore ' + str(fscore) + '\n')
            sys.stdout.flush()
            file_log.flush()

    