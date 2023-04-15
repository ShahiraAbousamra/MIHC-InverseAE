import torch.optim as optim
import os;
from distutils.util import strtobool;
import numpy as np;
import torch
import glob;
import skimage.io as io;

from ..sa_net_data_provider import AbstractDataProvider;
from ..sa_helpers import multiplex_utils ;


class MultiplexAutoencoderTesterRGB:
    def __init__(self, cnn_arch, cnn_arch_module, test_data_provider:AbstractDataProvider, session_config, output_dir, device, output_ext, kwargs):
        # Predefined list of arguments
        args = {'split_name':'test', 'batch_size':1, 'invert_out_img':'False', 'is_output_od':'False'};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        self.cnn_arch_module = cnn_arch_module;

        self.test_data_provider = test_data_provider;
        self.batch_size = int(args['batch_size']);

        self.invert_out_img = bool(strtobool(args['invert_out_img']));
        self.is_output_od = bool(strtobool(args['is_output_od']));

        self.device = device;
        self.output_dir = output_dir;
        self.output_ext = output_ext;

        if(not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)


    def test(self, do_init, do_restore, do_load_data):
        if(do_restore):
            checkpoint = self.cnn_arch_module.restore_model(is_test=True);            
            if(checkpoint is not None):
                print('restore succeed')

        if(do_load_data):
            self.test_data_provider.load_data();

        self.indx = 0;
        with torch.no_grad():            
            batch_x, batch_label, batch_filenames = self.test_data_provider.get_next_n(self.batch_size);
            total_count = 0;
            while(batch_x is not None):
                if(self.device is not None):
                    batch_x = batch_x.to(self.device);
                    batch_label = batch_label.to(self.device);
                batch_y, batch_concentration, batch_concentration_sm, batch_stains = self.cnn_arch(batch_x);

                batch_count = batch_label.shape[0];
                total_count += batch_count ;

                self.output_sample_results_concentrations_only(batch_label.detach().cpu().numpy()
                    , batch_y.detach().cpu().numpy()
                    , batch_concentration.detach().cpu().numpy()
                    , batch_concentration_sm.detach().cpu().numpy()
                    , batch_stains.detach().cpu().numpy()[0:3,:]
                    , batch_filenames
                );

                # Next batch
                batch_x, batch_label, batch_filenames = self.test_data_provider.get_next_n(self.batch_size);

     

  
    def output_stains(self, epoch, stains_matrix_od):
        stains_matrix_od = stains_matrix_od[0:3,:].transpose();
        stains_matrix_od.dump(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_od_epoch_'+str(epoch)+'.npy'));
        self.write_to_file("stains_matrix_od = " , self.epoch_out_filewriter);
        self.write_to_file(str(stains_matrix_od) , self.epoch_out_filewriter);
        stains_matrix_rgb = multiplex_utils.transform_optical_density_to_intensity(stains_matrix_od).astype(np.uint8);
        stains_matrix_rgb.dump(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_rgb_epoch_'+str(epoch)+'.npy'));
        self.write_to_file("stains_matrix_rgb = " , self.epoch_out_filewriter);
        self.write_to_file(str(stains_matrix_rgb) , self.epoch_out_filewriter);
        # Save stains visualization image
        square_side = 32;
        stains_visualize = np.zeros((square_side, stains_matrix_rgb.shape[0]*square_side, 3), dtype=np.uint8);   
        for i in range(stains_matrix_rgb.shape[0]):
            stains_visualize[:, i*square_side:(i+1)*square_side] =  stains_matrix_rgb[i];
        if(self.invert_out_img):
            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_vis_epoch_'+str(epoch)+'.png'), 255-stains_visualize);
        else:
            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_vis_epoch_'+str(epoch)+'.png'), stains_visualize);



    def output_sample_results_maxonly(self, batch_x, batch_y, batch_concentrations, batch_concentrations_sm, stains, batch_filenames):
        batch_x_rgb = batch_x.astype(np.uint8);
        batch_x_rgb = np.transpose(batch_x_rgb, axes=(0,2,3,1))
        batch_y_rgb = multiplex_utils.transform_optical_density_to_intensity(batch_y).astype(np.uint8);
        batch_y_rgb = np.transpose(batch_y_rgb, axes=(0,2,3,1))
        s_sum = stains.sum(axis=0)
        s_sum_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_sum,axis=0),axis=2),axis=3)
        b=np.multiply(s_sum_expand,batch_concentrations)

        s_max = stains.max(axis=0)
        s_max_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_max,axis=0),axis=2),axis=3)
        b_max = np.multiply(s_max_expand,batch_concentrations)

        s_mean = stains.mean(axis=0)
        s_mean_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_mean,axis=0),axis=2),axis=3)
        b_mean = np.multiply(s_mean_expand,batch_concentrations)

        for i in range(batch_x_rgb.shape[0]):
            self.indx += 1;
            img_filename = os.path.splitext(os.path.split(batch_filenames[i])[1])[0];

            batch_concentrations[i].dump(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+ '_conc'+'.npy'));
        
            batch_concentrations_flatten = batch_concentrations[i].reshape((batch_concentrations.shape[1], batch_concentrations.shape[2]*batch_concentrations.shape[3]));
            batch_concentrations_flatten_argmax = batch_concentrations_flatten.argmax(axis=0);
            one_hot =  np.zeros(batch_concentrations_flatten.shape);

            b_flatten = b[i].reshape((b.shape[1], b.shape[2]*b.shape[3]));
            b_flatten_argmax = b_flatten.argmax(axis=0);
            b_one_hot =  np.zeros(b_flatten.shape);

            b_mean_flatten = b_mean[i].reshape((b_mean.shape[1], b_mean.shape[2]*b_mean.shape[3]));
            b_mean_flatten_argmax = b_mean_flatten.argmax(axis=0);
            b_mean_one_hot =  np.zeros(b_mean_flatten.shape);

            one_hot[batch_concentrations_flatten_argmax,np.arange(one_hot.shape[1])]=1;
            one_hot_sum = one_hot.sum(axis=1);

            b_one_hot[b_flatten_argmax,np.arange(b_one_hot.shape[1])]=1;

            b_mean_one_hot[b_mean_flatten_argmax,np.arange(b_mean_one_hot.shape[1])]=1;

            new_concentration_flatten = one_hot * batch_concentrations_flatten;
            b_new_concentration_flatten = b_one_hot * batch_concentrations_flatten;
            b_mean_new_concentration_flatten = b_mean_one_hot * batch_concentrations_flatten;
            new_concentration = new_concentration_flatten.reshape((batch_concentrations.shape[1], batch_concentrations.shape[2], batch_concentrations.shape[3]));
            new_out_od = np.matmul(stains, new_concentration_flatten);
            new_out_od_mask = np.matmul(stains, one_hot);

            b_new_concentration = b_new_concentration_flatten.reshape((batch_concentrations.shape[1], batch_concentrations.shape[2], batch_concentrations.shape[3]));
            b_new_out_od = np.matmul(stains, b_new_concentration_flatten);
            b_new_out_od_mask = np.matmul(stains, b_one_hot);

            b_mean_new_concentration = b_mean_new_concentration_flatten.reshape((batch_concentrations.shape[1], batch_concentrations.shape[2], batch_concentrations.shape[3]));
            b_mean_new_out_od = np.matmul(stains, b_mean_new_concentration_flatten);

            new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(new_out_od).astype(np.uint8);
            new_out_rgb = new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            new_out_rgb = np.transpose(new_out_rgb, axes=(1,2,0))

            new_out_rgb_mask = multiplex_utils.transform_optical_density_to_intensity(new_out_od_mask).astype(np.uint8);
            new_out_rgb_mask = new_out_rgb_mask.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            new_out_rgb_mask = np.transpose(new_out_rgb_mask, axes=(1,2,0))

            b_new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(b_new_out_od).astype(np.uint8);
            b_new_out_rgb = b_new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            b_new_out_rgb = np.transpose(b_new_out_rgb, axes=(1,2,0))

            b_new_out_rgb_mask = multiplex_utils.transform_optical_density_to_intensity(b_new_out_od_mask).astype(np.uint8);
            b_new_out_rgb_mask = b_new_out_rgb_mask.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            b_new_out_rgb_mask = np.transpose(b_new_out_rgb_mask, axes=(1,2,0))

            b_mean_new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(b_mean_new_out_od).astype(np.uint8);
            b_mean_new_out_rgb = b_mean_new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            b_mean_new_out_rgb = np.transpose(b_mean_new_out_rgb, axes=(1,2,0))

            # Save each image and corresponding reconstruction in batch
            if(self.invert_out_img):
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'.png'), 255-batch_x_rgb[i]);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out.png'), 255-batch_y_rgb[i]);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly.png'), 255-new_out_rgb);
            else:
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'.png'), batch_x_rgb[i]);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out.png'), batch_y_rgb[i]);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly.png'), new_out_rgb);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly_b.png'), b_new_out_rgb);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly_mask.png'), new_out_rgb_mask);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly_b_mask.png'), b_new_out_rgb_mask);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly_b_mean_mask.png'), b_mean_new_out_rgb);


            # Save each stain estimate image
            for s in range(stains.shape[1]):
                stain_conc_img_od = np.matmul(stains[:,s].reshape((-1,1)), batch_concentrations[i][s].reshape(1,-1));
                stain_conc_img_od_maxonly = np.matmul(stains[:,s].reshape((-1,1)), new_concentration[s].reshape(1,-1));
                b_stain_conc_img_od_maxonly = np.matmul(stains[:,s].reshape((-1,1)), b_new_concentration[s].reshape(1,-1));
                stain_conc_img_od = np.transpose(stain_conc_img_od, axes=(1,0))
                stain_conc_img_od_maxonly = np.transpose(stain_conc_img_od_maxonly, axes=(1,0))
                b_stain_conc_img_od_maxonly = np.transpose(b_stain_conc_img_od_maxonly, axes=(1,0))
                stain_conc_img_od = stain_conc_img_od.reshape((batch_y_rgb.shape[1],batch_y_rgb.shape[2],3))
                stain_conc_img_od_maxonly = stain_conc_img_od_maxonly.reshape((batch_y_rgb.shape[1],batch_y_rgb.shape[2],3))
                b_stain_conc_img_od_maxonly = b_stain_conc_img_od_maxonly.reshape((batch_y_rgb.shape[1],batch_y_rgb.shape[2],3))
                stain_conc_img_rgb = multiplex_utils.transform_optical_density_to_intensity(stain_conc_img_od).astype(np.uint8);
                stain_conc_img_rgb_maxonly = multiplex_utils.transform_optical_density_to_intensity(stain_conc_img_od_maxonly).astype(np.uint8);
                b_stain_conc_img_rgb_maxonly = multiplex_utils.transform_optical_density_to_intensity(b_stain_conc_img_od_maxonly).astype(np.uint8);
                if(self.invert_out_img):
                    io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+ '_stain_'+ str(s)+'_out.png'), 255-stain_conc_img_rgb);
                    io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+ '_stain_'+ str(s)+'_out_maxonly.png'), 255-stain_conc_img_rgb_maxonly);
                else:
                    io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+ '_stain_'+ str(s)+'_out.png'), stain_conc_img_rgb);
                    io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+ '_stain_'+ str(s)+'_out_maxonly.png'), stain_conc_img_rgb_maxonly);
                    io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+ '_stain_'+ str(s)+'_out_maxonly_b.png'), b_stain_conc_img_rgb_maxonly);


    def output_sample_results_maxonly_noH(self, batch_x, batch_y, batch_concentrations, batch_concentrations_sm, stains, batch_filenames):

        batch_concentrations[:,6,:,:] = 0;

        batch_x_rgb = batch_x.astype(np.uint8);
        batch_x_rgb = np.transpose(batch_x_rgb, axes=(0,2,3,1))
        batch_y_rgb = multiplex_utils.transform_optical_density_to_intensity(batch_y).astype(np.uint8);
        batch_y_rgb = np.transpose(batch_y_rgb, axes=(0,2,3,1))
        s_sum = stains.sum(axis=0)
        s_sum_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_sum,axis=0),axis=2),axis=3)
        b=np.multiply(s_sum_expand,batch_concentrations)

        s_max = stains.max(axis=0)
        s_max_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_max,axis=0),axis=2),axis=3)
        b_max = np.multiply(s_max_expand,batch_concentrations)


        for i in range(batch_x_rgb.shape[0]):
            self.indx += 1;
            img_filename = os.path.splitext(os.path.split(batch_filenames[i])[1])[0];

            batch_concentrations[i].dump(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+ '_conc'+'.npy'));
        
            batch_concentrations_flatten = batch_concentrations[i].reshape((batch_concentrations.shape[1], batch_concentrations.shape[2]*batch_concentrations.shape[3]));
            batch_concentrations_flatten_argmax = batch_concentrations_flatten.argmax(axis=0);
            one_hot =  np.zeros(batch_concentrations_flatten.shape);

            b_flatten = b[i].reshape((b.shape[1], b.shape[2]*b.shape[3]));
            b_flatten_argmax = b_flatten.argmax(axis=0);
            b_one_hot =  np.zeros(b_flatten.shape);

            one_hot[batch_concentrations_flatten_argmax,np.arange(one_hot.shape[1])]=1;
            one_hot_sum = one_hot.sum(axis=1);

            b_one_hot[b_flatten_argmax,np.arange(b_one_hot.shape[1])]=1;

            new_concentration_flatten = one_hot * batch_concentrations_flatten;
            b_new_concentration_flatten = b_one_hot * batch_concentrations_flatten;
            new_concentration = new_concentration_flatten.reshape((batch_concentrations.shape[1], batch_concentrations.shape[2], batch_concentrations.shape[3]));
            new_out_od = np.matmul(stains, new_concentration_flatten);
            new_out_od_mask = np.matmul(stains, one_hot);

            b_new_concentration = b_new_concentration_flatten.reshape((batch_concentrations.shape[1], batch_concentrations.shape[2], batch_concentrations.shape[3]));
            b_new_out_od = np.matmul(stains, b_new_concentration_flatten);
            b_new_out_od_mask = np.matmul(stains, b_one_hot);

            new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(new_out_od).astype(np.uint8);
            new_out_rgb = new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            new_out_rgb = np.transpose(new_out_rgb, axes=(1,2,0))

            new_out_rgb_mask = multiplex_utils.transform_optical_density_to_intensity(new_out_od_mask).astype(np.uint8);
            new_out_rgb_mask = new_out_rgb_mask.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            new_out_rgb_mask = np.transpose(new_out_rgb_mask, axes=(1,2,0))

            b_new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(b_new_out_od).astype(np.uint8);
            b_new_out_rgb = b_new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            b_new_out_rgb = np.transpose(b_new_out_rgb, axes=(1,2,0))

            b_new_out_rgb_mask = multiplex_utils.transform_optical_density_to_intensity(b_new_out_od_mask).astype(np.uint8);
            b_new_out_rgb_mask = b_new_out_rgb_mask.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            b_new_out_rgb_mask = np.transpose(b_new_out_rgb_mask, axes=(1,2,0))

            # Save each image and corresponding reconstruction in batch
            if(self.invert_out_img):
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'.png'), 255-batch_x_rgb[i]);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out.png'), 255-batch_y_rgb[i]);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly.png'), 255-new_out_rgb);
            else:
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'.png'), batch_x_rgb[i]);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out.png'), batch_y_rgb[i]);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly.png'), new_out_rgb);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly_b.png'), b_new_out_rgb);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly_mask.png'), new_out_rgb_mask);
                io.imsave(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+'_out_maxonly_b_mask.png'), b_new_out_rgb_mask);


    def output_sample_results_concentrations_only(self, batch_x, batch_y, batch_concentrations, batch_concentrations_sm, stains, batch_filenames):

        for i in range(batch_concentrations.shape[0]):
            self.indx += 1;
            img_filename = os.path.splitext(os.path.split(batch_filenames[i])[1])[0];
            print(img_filename)

            batch_concentrations[i].astype(np.float16).dump(os.path.join(self.output_dir, self.cnn_arch_module.model_base_filename + '_'+img_filename+ '.npy'));
        

