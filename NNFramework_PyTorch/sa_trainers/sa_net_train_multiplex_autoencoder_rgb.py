#import tensorflow as tf;
import torch.optim as optim
import os;
from distutils.util import strtobool;
import numpy as np;
import torch
import glob;
import skimage.io as io;

from ..sa_net_train import CNNTrainer;
from ..sa_net_optimizer_pytorch import OptimizerTypesPyTorch, CNNOptimizerPyTorch;
from ..sa_net_data_provider import AbstractDataProvider;
from ..sa_helpers import multiplex_utils;


class MultiplexAutoencoderTrainerRGB(CNNTrainer):
    def __init__(self, cnn_arch, cnn_arch_module, train_data_provider:AbstractDataProvider, validate_data_provider:AbstractDataProvider, optimizer_type, session_config, device, kwargs):
        # Predefined list of arguments
        args = {'max_epochs':1000, 'learning_rate': 0.0005, 'batch_size':256, 'epoch_size':10, 'display_step':5, 'save_best_only':False
            , 'with_features':'False', 'invert_out_img':'False', 'is_output_od':'False','validate_step':10, 'wait_epochs':100};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        self.cnn_arch_module = cnn_arch_module;
        self.train_data_provider = train_data_provider;
        self.validate_data_provider = validate_data_provider;
        self.optimizer_type = optimizer_type;
        self.device = device;
        self.max_epochs = int(args['max_epochs']);
        self.learning_rate = float(args['learning_rate']);
        self.batch_size = int(args['batch_size']);
        self.epoch_size = int(args['epoch_size']);
        self.display_step = int(args['display_step']);
        self.save_best_only = bool(strtobool(args['save_best_only']));
        self.with_features = bool(strtobool(args['with_features']));
        self.invert_out_img = bool(strtobool(args['invert_out_img']));
        self.is_output_od = bool(strtobool(args['is_output_od']));
        self.validate_step = int(args['validate_step']);
        self.wait_epochs = int(args['wait_epochs']);
        print('self.save_best_only = {}'.format(self.save_best_only) );

        if(self.optimizer_type == OptimizerTypesPyTorch.ADAM):
            self.optimizer = CNNOptimizerPyTorch.adam_optimizer(self.learning_rate, self.cnn_arch, kwargs);
        elif(self.optimizer_type == OptimizerTypesPyTorch.SGD):
            self.optimizer = CNNOptimizerPyTorch.sgd_optimizer(self.learning_rate, self.cnn_arch, kwargs);

        self.epoch_out_filename = os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_train_epoch_out.txt');
        self.minibatch_out_filename = os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_train_minibatch_out.txt');

    def train(self, do_init, do_restore, do_load_data):
        self.epoch_out_filewriter = open(self.epoch_out_filename, 'a+' );
        self.minibatch_out_filewriter = open(self.minibatch_out_filename, 'a+' );

        epoch_start_num = 1;
        if(do_restore):
            checkpoint = self.cnn_arch_module.restore_model(is_test=False);            
            if(checkpoint is not None):
                print('restore succeed')
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']);
                epoch_start_num = checkpoint['epoch'] + 1;

        if(do_load_data):
            self.train_data_provider.load_data();
            if(not (self.validate_data_provider is None)):
                self.validate_data_provider.load_data();

        if(self.epoch_size < 0)    :
            self.epoch_size = int(self.train_data_provider.data_count / self.batch_size + 0.5);
        if(not(self.validate_data_provider is None)):
            self.validate_epoch_size = int(self.validate_data_provider.data_count );


        best_loss_saved_model_filename = None;
        last_saved_model_filename = None;

        best_validation_loss = float('inf')
        best_train_val_avg_accuracy = 0;
        best_validation_epoch = -1
        elapsed_wait_epochs=0

        current_validation_accuracy = None;    
        current_train_val_avg_accuracy = None;    

        for epoch in range(epoch_start_num, self.max_epochs+1):
            # Run in training mode to ensure batch norm is calculated based on running mean and std
            self.cnn_arch_module.change_mode(is_test=False);
            total_loss = 0;
            total_count = 0;

            for step in range(0, self.epoch_size):
                batch_correct_count = 0;
                self.cnn_arch.zero_grad();
                self.optimizer.zero_grad();
                # Get next batch. 
                batch_x, batch_label = self.train_data_provider.get_next_n(self.batch_size);
                if(batch_x is None):
                    break;
                # Move data to GPU
                if(self.device is not None):
                    batch_x = batch_x.to(self.device);
                    batch_label = batch_label.to(self.device);
                # Pass input to model and get prediction/estimation of reconstructed image in OD, raw stain concentration maps, and stain color vectors
                batch_y, batch_concentration, batch_concentration_sm, batch_stains = self.cnn_arch(batch_x);

                # Compute the loss
                loss = self.cnn_arch_module.cost_func.calc_cost(batch_y, batch_label, batch_concentration, batch_concentration_sm, batch_stains, self.device);

                # Backpropagate
                loss.backward();
                self.optimizer.step()


                batch_count = batch_label.shape[0];
                total_count += batch_count ;

                if step % self.display_step == 0:
                    print('step = {}'.format(step));
                    self.output_minibatch_info(epoch, step, loss, batch_count)
                        
                total_loss += loss.item();

                del batch_x, batch_label,batch_y, batch_concentration, batch_concentration_sm, batch_stains
            self.output_epoch_info(epoch, total_loss, self.epoch_size, total_count);                        
            
            elapsed_wait_epochs += 1
            if(epoch % self.validate_step == 0):
                # Validate
                if(not(self.validate_data_provider is None)):
                    # Run in test mode to ensure batch norm is calculated based on saved mean and std
                    self.cnn_arch_module.change_mode(is_test=True);
                    print("Running Validation:");
                    self.write_to_file("Running Validation"
                        , self.epoch_out_filewriter
                    );
                    self.validate_data_provider.reset();
                    validate_total_loss = 0;
                    validate_count = 0;
                    for validate_step in range(0, self.validate_epoch_size):
                        validate_batch_x, validate_batch_label = self.validate_data_provider.get_next_n(1);
                        if(validate_batch_x is None):
                            break;
                        if(self.device is not None):
                            validate_batch_x = validate_batch_x.to(self.device) ;
                            validate_batch_label = validate_batch_label.to(self.device) ;
                        validate_batch_y, validate_batch_concentration, validate_batch_concentration_sm, validate_batch_stains = self.cnn_arch(validate_batch_x);
                        validate_loss = self.cnn_arch_module.cost_func.calc_cost(validate_batch_y, validate_batch_label, validate_batch_concentration, validate_batch_concentration_sm, validate_batch_stains, self.device);
                        validate_total_loss += validate_loss.item();
                        validate_count += validate_batch_y.shape[0] ;
                        if(validate_step < 6):
                            self.output_stains(epoch, validate_batch_stains.detach().cpu().numpy()[:,:]);


                            if(self.is_output_od):
                                self.output_sample_results_maxonly(epoch, validate_batch_label.detach().cpu().numpy()
                                    , validate_batch_y.detach().cpu().numpy()
                                    , validate_batch_concentration.detach().cpu().numpy()
                                    , validate_batch_concentration_sm.detach().cpu().numpy()
                                    , validate_batch_stains.detach().cpu().numpy()[:,:]
                                    , validate_step
                                );
                            else:
                                self.output_sample_results_rgb(epoch, validate_batch_label.detach().cpu().numpy()
                                    , validate_batch_y.detach().cpu().numpy()
                                    , validate_batch_concentration_sm.detach().cpu().numpy()
                                    , validate_batch_stains.detach().cpu().numpy()[0:3,:]
                                );
                            del validate_batch_x, validate_batch_label, validate_batch_concentration, validate_batch_concentration_sm, validate_batch_y
                    

                    self.output_epoch_info(epoch, validate_total_loss, validate_step+1, validate_count);                        
            
            
                saved = False;

                if(self.save_best_only):
                    if(validate_total_loss < best_validation_loss) :
                        best_validation_loss = validate_total_loss;
                        print("Saving model:");
                        new_best_saved_model_filename = self.cnn_arch_module.save_model(None, self.optimizer, epoch, suffix="_loss");
                        self.delete_model_files(best_loss_saved_model_filename);
                        best_loss_saved_model_filename = new_best_saved_model_filename;
                        best_validation_epoch = epoch
                        saved = True;
                        elapsed_wait_epochs =0
                    if(not saved):
                        new_saved_model_filename = self.cnn_arch_module.save_model(None, self.optimizer, epoch);
                        self.delete_model_files(last_saved_model_filename);
                        last_saved_model_filename = new_saved_model_filename;
                else:
                    new_saved_model_filename = self.cnn_arch_module.save_model(None, self.optimizer, epoch);
                    last_saved_model_filename = new_saved_model_filename;
                    if(validate_total_loss < best_validation_loss) :
                        best_validation_loss = validate_total_loss;
                        best_loss_saved_model_filename = new_saved_model_filename
                        best_validation_epoch = epoch
                        elapsed_wait_epochs = 0
                if(elapsed_wait_epochs > self.wait_epochs):
                    elapsed_wait_epochs = 0
                    self.write_to_file("reduce learning rate to " + str(self.learning_rate/2) \
                        , self.minibatch_out_filewriter
                    );
                    self.write_to_file('fall back to epoch ' + str(epoch) \
                        , self.minibatch_out_filewriter
                    );
                    checkpoint = self.cnn_arch_module.restore_model(is_test=False, model_restore_filename=best_loss_saved_model_filename);                                
                    if(checkpoint is not None):
                        self.learning_rate /= 2
                        if(self.optimizer_type == OptimizerTypesPyTorch.ADAM):
                            self.optimizer = CNNOptimizerPyTorch.adam_optimizer(self.learning_rate, self.cnn_arch);
                        elif(self.optimizer_type == OptimizerTypesPyTorch.SGD):
                            self.optimizer = CNNOptimizerPyTorch.sgd_optimizer(self.learning_rate, self.cnn_arch);
                        self.write_to_file('restore succeed'  \
                            , self.minibatch_out_filewriter
                        );
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']);


            # Permute the training data for the next epoch
            self.train_data_provider.reset(repermute=True);
            
        
        self.epoch_out_filewriter.close();
        self.minibatch_out_filewriter.close();


    def output_minibatch_info(self, epoch, batch, cost, total_count):
        print("epoch = " + str(epoch) \
            + ", batch# = " + str(batch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", total count= " + "{:d}".format(total_count) \
        );
        self.write_to_file("epoch = " + str(epoch) \
            + ", batch# = " + str(batch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", total count= " + "{:d}".format(total_count) \
            , self.minibatch_out_filewriter
        );


    def output_epoch_info(self, epoch, total_cost, n_batches, total_count):
        print("\r\nepoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
        );
        self.write_to_file("epoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
            , self.epoch_out_filewriter
        );
        self.write_to_file("\r\n epoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", total count= " + "{:d}".format(total_count) \
            , self.minibatch_out_filewriter
        );

    def output_stains(self, epoch, stains_matrix_od):
        stains_matrix_od = stains_matrix_od.transpose();
        stains_matrix_od.dump(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_stains_od_epoch_'+str(epoch)+'.npy'));
        print("stains_matrix_od = ");
        print(stains_matrix_od);
        self.write_to_file("stains_matrix_od = " , self.epoch_out_filewriter);
        self.write_to_file(str(stains_matrix_od) , self.epoch_out_filewriter);
        stains_matrix_rgb = multiplex_utils.transform_optical_density_to_intensity(stains_matrix_od).astype(np.uint8);
        if(stains_matrix_rgb.shape[1] == 4):
            stains_matrix_rgb = multiplex_utils.transform_cmyk_to_rgb_1D(stains_matrix_rgb);
        print("stains_matrix_rgb = ");
        print(stains_matrix_rgb);
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



    def output_sample_results_maxonly(self, epoch, batch_x, batch_y, batch_concentrations, batch_concentrations_sm, stains, validate_step):
        const_val = 255.0;
        batch_x_rgb = batch_x.astype(np.uint8);
        if(batch_x_rgb.shape[1]<20):
            batch_x_rgb = np.transpose(batch_x_rgb, axes=(0,2,3,1))
        if(batch_x_rgb.shape[-1] == 4):
            for i in range(batch_x_rgb.shape[0]):
                batch_x_rgb[i,:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(batch_x_rgb[i]);
                batch_x_rgb[i,:,:,3] = 255
        elif(batch_x_rgb.shape[-1]>4):
            batch_x_rgb = batch_x_rgb[:,:,:,:3];
        batch_y_rgb = multiplex_utils.transform_optical_density_to_intensity(batch_y, const_val=const_val).astype(np.uint8);
        batch_y_rgb = np.transpose(batch_y_rgb, axes=(0,2,3,1))
        if(batch_y_rgb.shape[-1] == 4):
            for i in range(batch_x_rgb.shape[0]):
                batch_y_rgb[i,:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(batch_y_rgb[i]); 
                batch_y_rgb[i,:,:,3] = 255;
        s_sum = stains.sum(axis=0)
        s_sum_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_sum,axis=0),axis=2),axis=3)
        b=np.multiply(s_sum_expand,batch_concentrations)

        s_max = stains.max(axis=0)
        s_max_expand = np.expand_dims(np.expand_dims(np.expand_dims(s_max,axis=0),axis=2),axis=3)
        b_max = np.multiply(s_max_expand,batch_concentrations)

        for i in range(min(batch_x_rgb.shape[0], 5)):
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

            new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(new_out_od, const_val=const_val).astype(np.uint8);
            new_out_rgb = new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            new_out_rgb = np.transpose(new_out_rgb, axes=(1,2,0))
            if(new_out_rgb.shape[-1] == 4):
                new_out_rgb[:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(new_out_rgb);
                new_out_rgb[:,:,3]  = 255;

            new_out_rgb_mask = multiplex_utils.transform_optical_density_to_intensity(new_out_od_mask, const_val=const_val).astype(np.uint8);
            new_out_rgb_mask = new_out_rgb_mask.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            new_out_rgb_mask = np.transpose(new_out_rgb_mask, axes=(1,2,0))
            if(new_out_rgb_mask.shape[-1] == 4):
                new_out_rgb_mask[:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(new_out_rgb_mask);
                new_out_rgb_mask[:,:,3]  = 255;

            b_new_out_rgb = multiplex_utils.transform_optical_density_to_intensity(b_new_out_od, const_val=const_val).astype(np.uint8);
            b_new_out_rgb = b_new_out_rgb.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            b_new_out_rgb = np.transpose(b_new_out_rgb, axes=(1,2,0))
            if(b_new_out_rgb.shape[-1] == 4):
                b_new_out_rgb[:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(b_new_out_rgb);
                b_new_out_rgb[:,:,3]  = 255;

            b_new_out_rgb_mask = multiplex_utils.transform_optical_density_to_intensity(b_new_out_od_mask, const_val=const_val).astype(np.uint8);
            b_new_out_rgb_mask = b_new_out_rgb_mask.reshape((batch_y.shape[1], batch_y.shape[2], batch_y.shape[3]));
            b_new_out_rgb_mask = np.transpose(b_new_out_rgb_mask, axes=(1,2,0))
            if(b_new_out_rgb_mask.shape[-1] == 4):
                b_new_out_rgb_mask[:,:,0:3] = multiplex_utils.transform_cmyk_to_rgb(b_new_out_rgb_mask);
                b_new_out_rgb_mask[:,:,3]  = 255;


            # Save each image and corresponding reconstruction in batch
            if(self.invert_out_img):
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'.png'), 255-batch_x_rgb[i]);
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'_out.png'), 255-batch_y_rgb[i]);
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'_out_maxonly.png'), 255-new_out_rgb);
            else:
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'.png'), batch_x_rgb[i]);
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'_out.png'), batch_y_rgb[i]);
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'_out_maxonly.png'), new_out_rgb);
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'_out_maxonly_b.png'), b_new_out_rgb);
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'_out_maxonly_mask.png'), new_out_rgb_mask);
                io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i+validate_step)+'_out_maxonly_b_mask.png'), b_new_out_rgb_mask);




    def output_sample_results_rgb(self, epoch, batch_x, batch_y, batch_concentrations, stains):
        batch_x_rgb = batch_x.astype(np.uint8);
        batch_x_rgb = np.transpose(batch_x_rgb, axes=(0,2,3,1))
        if(batch_x_rgb.shape[-1] < 3):
            batch_x_rgb = np.repeat(batch_x_rgb,3,axis=-1)
        batch_y_rgb = ((batch_y)).astype(np.uint8);
        batch_y_rgb = np.transpose(batch_y_rgb, axes=(0,2,3,1))
        if(batch_y_rgb.shape[-1] < 3):
            batch_y_rgb = np.repeat(batch_y_rgb,3,axis=-1)
        for i in range(min(batch_x_rgb.shape[0], 5)):
            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'.png'), batch_x_rgb[i]);
            io.imsave(os.path.join(self.cnn_arch_module.model_out_path, self.cnn_arch_module.model_base_filename + '_epoch_'+str(epoch) + '_sample_' + str(i)+'_out.png'), batch_y_rgb[i]);



    def write_to_file(self, text, filewriter):
        filewriter.write('\r\n');
        filewriter.write(text);
        filewriter.flush();

    def print_optimizer_params(self):
        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])




    def delete_model_files(self, filepath):
        if(filepath is None):
            return;
        filepath, _ = os.path.splitext(filepath);
        print('delete_model_files = ', filepath)
        file_pattern = filepath + '*';
        files = glob.glob(file_pattern);
        for file in files: 
            print(file);
            os.remove(file);

