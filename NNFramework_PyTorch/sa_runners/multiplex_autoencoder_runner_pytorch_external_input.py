import sys;
import os;
import configparser;
import torch;
#sys.path.append("..");

# The runner file parses the config file and instantiates the appropriate classes with the configured parameters.

def load_model(config_filepath):

    device_ids_str = None;
    # Read the gpu ids to use from the command line parameters if cuda is available
    if(not (device_ids_str is None)):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str;
        
    device_ids_str = os.environ["CUDA_VISIBLE_DEVICES"];
    print(device_ids_str);

    # Read the gpu ids to use from the command line parameters if cuda is available
    device_ids = [];
    device = None;
    if(torch.cuda.is_available()):
        # Get number of available cuda devices
        gpu_count = torch.cuda.device_count();
        # Create list of gpu ids, excluding invalid gpus
        if(not(device_ids_str is None)):
            device_ids_original = [int(g) for g in device_ids_str.split(",")];            
            device_ids = [j for j in range(len(device_ids_original))];            
            print(device_ids)
            i = 0;
            while(i < len(device_ids)):
                gpu_id = device_ids[i];
                if ((gpu_id >= gpu_count) or (gpu_id < 0)):
                    device_ids.remove(gpu_id);
                else:
                    i = i+1;
            print(device_ids)
            # Set the device where data will be placed as the first one in the list
            device = torch.device("cuda:"+str(device_ids[0]) if torch.cuda.is_available() else "cpu");
            print('device_ids[0]' + str(device_ids[0]));
   
    print('device_ids = ' + str(device_ids));
    # If no gpu then cpu
    if(device is None):
        device = torch.device("cpu");
    #device_ids = [1,2];    
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu");
   


    # read the config file
    config = configparser.ConfigParser();

    config.read(config_filepath);        
    # General config
    config_name = config['DEFAULT']['config_name'].strip();  ## strip: trims white space
    running_mode = config['DEFAULT']['mode'].strip();  
    if(running_mode == 'train'):
        is_test = False;
    else:
        is_test = True;
    model_path = config['DEFAULT']['model_path'].strip();
    print('model_path = '+ model_path);
    model_base_filename = config['DEFAULT']['model_base_filename'].strip();
    if('model_restore_filename' in config['DEFAULT']):
        model_restore_filename = config['DEFAULT']['model_restore_filename'].strip();
    else:
        model_restore_filename = None;

    # Network config
    network_params = dict(config.items('NETWORK'));
    network_class_name = config['NETWORK']['class_name'].strip();  
    n_channels = int(config['NETWORK']['n_channels'].strip());  
    n_classes = int(config['NETWORK']['n_classes'].strip());  

    # Cost config
    cost_params = dict(config.items('COST'));
    cost_func_class_name = config['COST']['class_name'].strip();  

    has_validation = False;
    if(is_test == False):
        # Train Data config
        train_params = dict(config.items('TRAIN_DATA'));
        train_config = config['TRAIN_DATA'];
        train_dataprovider_class_name = train_config['provider_class_name'].strip();  
        train_filepath_data = train_config['filepath_data'].strip();  
        train_filepath_label = None;
        if('filepath_label' in train_config):
            train_filepath_label = train_config['filepath_label'].strip();          
        train_preprocess = train_config.getboolean('preprocess');
        train_augment = train_config.getboolean('augment');
        train_permute = train_config.getboolean('permute');

        # Trainer config
        trainer_params = dict(config.items('TRAINER'));
        trainer_config = config['TRAINER'];
        trainer_class_name = trainer_config['class_name'].strip();  
        trainer_optimizer_type = trainer_config['optimizer_type'].strip();  

        # Validation Data config
        if('VALIDATE_DATA' in config):
            has_validation = True;
            validate_params = dict(config.items('VALIDATE_DATA'));
            validate_config = config['VALIDATE_DATA'];
            validate_dataprovider_class_name = validate_config['provider_class_name'].strip();  
            validate_filepath_data = validate_config['filepath_data'].strip();  
            validate_filepath_label = None;
            if('filepath_label' in validate_config):
                validate_filepath_label = validate_config['filepath_label'].strip();          
            validate_preprocess = validate_config.getboolean('preprocess');
            validate_augment = validate_config.getboolean('augment');
            validate_permute = validate_config.getboolean('permute');

    else:
        # Test Data config
        test_params = dict(config.items('TEST_DATA'));
        test_config = config['TEST_DATA'];
        test_dataprovider_class_name = test_config['provider_class_name'].strip();  
        test_filepath_data = test_config['filepath_data'].strip();  
        test_filepath_label = None;
        if('filepath_label' in test_config):
            test_filepath_label = test_config['filepath_label'].strip();          
        test_preprocess = test_config.getboolean('preprocess');
        test_augment = test_config.getboolean('augment');

        # Tester config
        tester_params = dict(config.items('TESTER'));
        tester_config = config['TESTER'];
        tester_class_name = tester_config['class_name'].strip();  
        tester_out_dir = tester_config['out_dir'].strip();  
        tester_out_ext = tester_config['out_ext'].strip();  
    

    # Configure Loss function
    if(cost_func_class_name == 'MSECostODWithInv'):
        from ..sa_cost_func.mse_cost_func_od_w_inv import MSECostODWithInv;
        cost_func = MSECostODWithInv(kwargs=cost_params);
    else:
        print('error: cost function class name \'{}\' is not supported by runner'.format(cost_func_class_name));
        sys.exit();        

    # Configure Network architecture
    if(network_class_name == 'MultiplexAutoencoderFixedStainsArch3Next3InputStain'):
        from ..sa_networks.multiplex_autoencoder_fixed_stains_arch3_next3_input_stain import MultiplexAutoencoderFixedStainsArch3Next3InputStain;
        cnn_arch = MultiplexAutoencoderFixedStainsArch3Next3InputStain(n_channels = n_channels, model_out_path = model_path, model_base_filename = model_base_filename, model_restore_filename = model_restore_filename, cost_func = cost_func \
            , device=device, kwargs=network_params \
            );
    else:
        print('error: network class name \'{}\' is not supported by runner'.format(network_class_name));
        sys.exit(); 
    print("distribute arch over gpus ")
    # Distribute arch over gpus  
    params = list(cnn_arch.parameters());  # The learnable parameters of a model are returned by net.parameters()
    print(len(params));  
    if (torch.cuda.device_count() > 1 and len(device_ids) > 1):
        cnn_arch = torch.nn.DataParallel(cnn_arch, device_ids);
    print("before move to device ", device)
    cnn_arch.to(device);
    print("after move to device")
    try:
        if(cnn_arch.module is None):
            cnn_arch_module  = cnn_arch;
        else:
            cnn_arch_module  = cnn_arch.module;    
    except:
        cnn_arch_module  = cnn_arch;

    if(is_test == False):
        if(train_dataprovider_class_name == 'MultiplexAutoencoderDataProviderRGB'):
            from sa_data_providers.multiplex_autoencoder_data_provider_rgb import MultiplexAutoencoderDataProviderRGB;
            train_data_provider = MultiplexAutoencoderDataProviderRGB( \
                is_test=is_test \
                , filepath_data = train_filepath_data \
                , filepath_label = train_filepath_label \
                , n_channels = n_channels \
                , n_classes = n_classes \
                , do_preprocess = train_preprocess \
                , do_augment = train_augment \
                , data_var_name = None \
                , label_var_name = None \
                , permute = train_permute \
                , repeat = True \
                , kwargs = train_params\
            );
        else:
            print('error: train data provider class name \'{}\' is not supported by runner'.format(train_dataprovider_class_name));
            sys.exit();        

        if(has_validation):
            if(validate_dataprovider_class_name == 'MultiplexAutoencoderDataProviderRGB'):
                from sa_data_providers.multiplex_autoencoder_data_provider_rgb import MultiplexAutoencoderDataProviderRGB;
                validate_data_provider = MultiplexAutoencoderDataProviderRGB( \
                    is_test=is_test \
                    , filepath_data = validate_filepath_data \
                    , filepath_label = validate_filepath_label \
                    , n_channels = n_channels \
                    , n_classes = n_classes \
                    , do_preprocess = validate_preprocess \
                    , do_augment = validate_augment \
                    , data_var_name = None \
                    , label_var_name = None \
                    , permute = validate_permute \
                    , repeat = False \
                    , kwargs = validate_params\
                );
            else:
                print('error: validate data provider class name \'{}\' is not supported by runner'.format(validate_dataprovider_class_name));
                sys.exit();  
        else:
            validate_data_provider = None;
    else:
        if(test_dataprovider_class_name == 'MultiplexAutoencoderDataProviderRGBTest'):
            from sa_data_providers.multiplex_autoencoder_data_provider_rgb_test import MultiplexAutoencoderDataProviderRGBTest;
            test_data_provider = MultiplexAutoencoderDataProviderRGBTest( \
                is_test=is_test \
                , filepath_data = test_filepath_data \
                , filepath_label = None \
                , n_channels = n_channels \
                , n_classes = n_classes \
                , do_preprocess = test_preprocess \
                , do_augment = test_augment \
                , data_var_name = None \
                , label_var_name = None \
                , permute = False \
                , repeat = False \
                , kwargs = test_params\
            );
        elif(not (test_dataprovider_class_name.lower() == 'none')):
            print('error: test data provider class name \'{}\' is not supported by runner'.format(test_dataprovider_class_name));
            sys.exit();         

    session_config = None;
    
    print("is_test = ", is_test)
    if(is_test == False):
        if(trainer_optimizer_type == 'ADAM'):
            optimizer_type=OptimizerTypesPyTorch.ADAM;
        elif(trainer_optimizer_type == 'SGD'):
            optimizer_type=OptimizerTypesPyTorch.SGD;
        else:
            print('error: trainer optimizer type \'{}\' is not supported by runner'.format(trainer_class_name));
            sys.exit();        
    
        if(tester_class_name == 'MultiplexAutoencoderTesterRGB'):
            from sa_testers.sa_net_test_mypytorch_multiplex_autoencoder import MultiplexAutoencoderTesterRGB;
            tester = MultiplexAutoencoderTesterRGB(cnn_arch \
                , cnn_arch_module \
                , test_data_provider \
                , session_config=session_config \
                , device = device \
                , output_dir=tester_out_dir \
                , output_ext=tester_out_ext \
                , kwargs = tester_params \
            );
        elif(tester_class_name == 'MultiplexAutoencoderTesterRGBExternalInput'):
            from ..sa_testers.sa_net_test_mypytorch_multiplex_autoencoder_external_input import MultiplexAutoencoderTesterRGBExternalInput;
            tester = MultiplexAutoencoderTesterRGBExternalInput(cnn_arch \
                , cnn_arch_module \
                , session_config=session_config \
                , device = device \
                , output_dir=tester_out_dir \
                , output_ext=tester_out_ext \
                , kwargs = tester_params \
            );
            tester.init_model(do_init=True, do_restore=True);
        else:
            print('error: tester class name \'{}\' is not supported by runner'.format(tester_class_name));
            sys.exit();        

    if(is_test == False):
        return trainer;
    else:
        return tester;
