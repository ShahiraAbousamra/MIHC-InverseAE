from .sa_net_data_provider import AbstractDataProvider;


class CNNTrainer:
    def __init__(self, cnn_arch, train_data_provider:AbstractDataProvider, optimizer_type, session_config, kwargs):
         pass;

    def train(self, do_init, do_restore, do_load_data, display_step=5):
         pass;


    def output_minibatch_info(self, epoch, cost, accuracy):
        print("epoch = " + str(epoch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", accuracy= " + "{:.6f}".format(accuracy) \
        );

    def output_epoch_info(self, epoch, total_cost):
        print("\r\nepoch = " + str(epoch) \
            + ", total loss= " + "{:.6f}".format(total_cost) \
        );
