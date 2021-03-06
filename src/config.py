modes = ["gen", "dis"]

# training settings
batch_size_gen = 8  # batch size for the generator
batch_size_dis = 8  # batch size for the discriminator
lambda_gen = 0.002  # l2 loss regulation weight for the generator
lambda_dis = 0.002  # l2 loss regulation weight for the discriminator
n_sample_gen = 40  # number of samples for the generator
lr_gen = 5e-3  # learning rate for the generator
lr_dis = 5e-3  # learning rate for the discriminator
n_epochs = 30000  # number of outer loops
n_epochs_gen = 50  # number of inner loops for the generator
n_epochs_dis = 50  # number of inner loops for the discriminator
gen_interval = 1    # n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
dis_interval = 1    # n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
temperature = 0.2

# model saving
load_model = False  # whether loading existing model for initialization
save_steps = 10

# other hyper-parameters
emb_dim = 16
missing_edge = 128
n_layers = 3
n_eigs = 6
test_interval = 1

log = "no GCN"

# path settings
train_filename = "../data/ml-1m/train_users.dat"
test_filename = "../data/ml-1m/test_users.dat"


def print_config():
    print("====================== config ======================")
    print("lr_gen {}, lr_dis {}".format(lr_gen, lr_dis))
    print("lambda_gen {}, lambda_dis {}".format(lambda_gen, lambda_dis))
    print("n_epochs_gen {}, n_epochs_dis {}".format(n_epochs_gen, n_epochs_dis))
    print("n_layers {}, n_eigs {}, missing_edge {}".format(n_layers, n_eigs, missing_edge))
    print("===================== note ========================\n", log)
