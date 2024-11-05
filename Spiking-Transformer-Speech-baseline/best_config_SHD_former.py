from spikingjelly.activation_based import surrogate
import torch.nn as nn
import datetime


class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False
    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    dataset = 'shd'
    datasets_path = '../Datasets/SHD'
    log_dir = './logs/logging/shd/'
    log_dir_test = './logs/logging/testing'

    seed = 312 # 312 42 3407 0 10086 114514+-5 3112
    gpu = 0
    model_type = 'spike-temporal-former'
    block_type = 'spikformer'

    distribute = False

    spike_mode = "lif"
    time_step = 10
    n_bins = 5

    epochs = 300
    n_warmup = 10


    batch_size = 256 # 128 => 256 => 512
    # dropout_l control the first layer
    dropout_l = 0.1
    # dropout_p control the layers in attention
    dropout_p = 0.1
    # MLP_RATIO
    mlp_ratio = 1
    #
    split_ratio = 2

    ############################
    #        USE Module        #
    ############################
    use_norm = False
    use_ln = False
    use_lif = False
    use_bn = True
    use_dp = True



    backend = 'torch'
    attn_mode = 'v2'
    depths = 1
    bias = True



    t_max = 40 # 40
    lr_w = 0.01 # 1e-3
    weight_decay = 0.01  # Default 1e-2 => 5e-3 => 1e-4
    n_inputs = 700//n_bins
    n_hidden_neurons_list =  [128] # 128,176,192,256

    n_outputs = 20 if dataset == 'shd' else 35

    num_heads = 8
    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum'
    loss_fn = 'CEloss'

    init_tau = 2.0 if spike_mode == "plif" else 2.0  # LIF
    v_threshold = 1.0 # LIF
    output_v_threshold = 2.0 if loss == 'spike_count' else 1e9  # use 1e9 for loss = 'mean' or 'max'
    gate_v_threshold = 1.0  # LIF
    alpha = 5.0

    surrogate_function = surrogate.ATan(alpha = alpha)
    detach_reset = True
    init_w_method = 'kaiming_uniform'
    max_len = 126
    use_padding = False
    norm_type = "bn"

    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adamw'
    optimizer_pos = 'adamw'


    ################################################
    #                    Save                      #
    ################################################
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = model_type
    run_info = f'||{dataset}||{depths}depths||{time_step}ms||bins={n_bins}||lr_w={lr_w}||heads={num_heads}'
    wandb_run_name = run_name + f'||seed={seed}' + run_info
    # # REPL is going to be replaced with best_acc or best_loss for best model according to validation accuracy or loss
    save_model_path = f'{wandb_run_name}_REPL_{current_time}.pt'
    make_plot = False
