# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    # "expname": "trex_hybrid/1debug_generator_s2g4k_allmsplus2mul_pl1e-2_dl3e-3_flr2e-2_dsdetach",#_originconf",
    # "expname": "trex_hybrid/1grad_gutecode3x_s2g4k_s2woocc_evalalign2e-3_mul_pl2e-2_dl3e-3_flr2e-2_dsdetach",#_originconf",
    # "expname": "trex_hybrid/1_dinograd_div3_alr5e-3_lr2e-3toall2e-2_block4_dlr2e-3_s2g4k_newpl2e-2to1e-3_nolinear",
    "expname": "trex_hybrid/keeplast10_sectest2",
    "logdir": "./logs/staticreal",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 4,
    "data_dirs": ["/media/public/disk5/lu/data/nerf_llff_data/trex"],
    
    # Data settings for LLFF
    "hold_every": 8,
    "contract": False,
    "ndc": True,
    "near_scaling": 0.89,
    "ndc_far": 2.6,  # for ndc

    # Pose settigs
    "learn_pose": True,
    "learn_R": True,
    "learn_t": True,
    "pose_lr": 5e-2,#2e-2,#1e-2, #5e-4,  #TODO: modify
    # "2stage_poselr": 5e-4,  # default 1e-3
    "pose_lr_end": 1e-5,
    # "t_lr": 1e-2,
    # "porf_lr": 5e-3,

    # Optimization settings
    "eval_steps": 10_01,
    "num_steps": 80_001, #60_001,
    "batch_size": 8192, #4096,
    "eval_batch_size": 4096, #4096,
    "code_mode": "mul",#"add",
    "code_mode_ds": "mul",#"add",
    "ms_res": False,
    # "code_mode_s2": "add",#"mul",
    # "code_mode_ds_s2": "add",#"mul",
    "scheduler_type": "warmup_cosine",
    "warmup_step": 128,#512, #FIXME:128
    "warmup_step_s2": 128,
    # "scheduler_type": "step",
    'scheduler_type_pose': "warmup_expdecay",
    "scheduler_type_pose_s2": "expdecay",
    "eval_scheduler_type": "step",
    "pose_warmup_steps": 128,#512,   # 1024
    "pose_warmup_steps_s2": 1024,   # 1024
    "optim_type": "adam",
    
    "switch2grids_step": 4001,#4001, #FIXME: 4001
    # "f_step": 500, #FIXME:300, 
    "field_lr": 3e-2,#[2e-2, 2e-2, 2e-2, 2e-2],#[2e-2,1e-2,5e-3,3e-3],#[1e-2, 1e-2, 5e-3, 5e-3],#[2e-2, 2e-2, 1e-2, 1e-2],   # TODO: 
    "lr": 2e-3,# 3e-3,#1e-2,#2e-2,
    "lr_s2": 3e-2,
    "attn_lr": 5e-3,# 3e-3,#1e-2,#2e-2,
    "density_lr": 2e-3,#1e-2,#1e-2,#5e-3,#1e-2,
    "eval_pose_lr": 2e-3,#1e-2,

    # Regularization
    "plane_tv_weight": 1e-4,
    "plane_tv_weight_proposal_net": 1e-4,
    "l1_proposal_net_weight": 0,
    "histogram_loss_weight": 1.0, 
    "distortion_loss_weight": 0.001,

    # Training settings
    "train_fp16": False,#True,
    "render_every": -1,#1_000,
    "save_every": 10_000,#20_000,
    "valid_every": 10_000,#0,#20_000,  #TODO: debug use
    "save_outputs": True,

    # Raymarching settings
    "num_samples": 48,
    # "num_uniform_samples": 128,
    "single_jitter": False,
    # proposal sampling
    "num_proposal_samples": [256, 128],
    "num_proposal_iterations": 2,
    "use_same_proposal_network": False,
    "use_proposal_weight_anneal": True,
    "proposal_net_args_list": [
        {"resolution": [128, 128, 128], "num_input_coords": 3, "num_output_coords": 8},  # origin kp: "num_output_coords": 8
        {"resolution": [256, 256, 256], "num_input_coords": 3, "num_output_coords": 8},  # "num_output_coords": 8
        # {"resolution": [64, 64, 64], "num_input_coords": 3, "num_output_coords": 16},  # origin kp: "num_output_coords": 8
        # {"resolution": [128, 128, 128], "num_input_coords": 3, "num_output_coords": 16},  # "num_output_coords": 8
    ],
    'proposal_generator_args_list': [
        {'model_ch': 8, 'model_out_ch': 8, 'model_res': [32, 32, 32], 'up_block_num': 3, 'block_out_channels': (32,64, 128)},
        {'model_ch': 8, 'model_out_ch': 8, 'model_res': [64, 64, 64], 'up_block_num': 3, 'block_out_channels': (32,64, 128)}    #TODO:
    ],

    # Model settings
    "tensor_config": ['xy', 'xz', 'yz'],#['xy', 'yz', 'zx'], @remind 和kplanes grids的顺序一致
    "multiscale_res": [1, 2, 4, 8],
    "model_ch": 16,#8,
    "model_out_ch": 64,#16,
    "model_out_ch_": 64,
    "model_res": [20, 20, 20],
    'model_up_block_num': 4,#5,
    'model_block_out_channels': (64, 64, 128, 256), #(32, 64, 64, 128, 256),
    
    "pose_ch": 8,
    "pose_out_ch": 16,
    "pose_res": [8,8,8],#[20,20,20],
    'pose_up_block_num': 3,
    'pose_block_out_channels': (32, 64, 128), #(32, 64, 64, 128, 256),
    
    # "scale_step": [8000, 15000, 20000, 25000],
    # "scale_step": [3000, 6000, 9000],#[9000, 18000, 27000],
    "density_activation": "trunc_exp",
    "concat_features_across_scales": True,
    "linear_decoder": False,
    "grid_config": [{
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 16,
        "grid_dimensions": 2,
        "resolution": [64, 64, 64],
    }],
    
}
