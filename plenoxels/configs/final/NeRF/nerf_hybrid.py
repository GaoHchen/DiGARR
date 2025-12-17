config = {
#  'expname': 'lego_hybrid/flip_noise10_randbg_lr2e-3_plr1e-2_dslr2e-3_ch16block2',
#  'expname': 'lego_hybrid/1bg_ns20_pl2e-2_lr1e-3_dl2e-3',#notrand_ns20_linear_plr1e-2_4k_eval4_lr1e-3_1001',
#  'expname': 'lego_hybrid/29.67_bg_ns15_pl2e-2_lr1e-3_dl2e-3/eval_ds4_pl5e-4',#notrand_ns20_linear_plr1e-2_4k_eval4_lr1e-3_1001',
#  'expname': 'lego_hybrid/AA_evalalign/bg1ts_6w_bgallrand_ns20_linear_plr2e-2_4k',
  # "expname": 'lego_hybrid/1grad_ns20_trbgrand_allcode3x_s2g2k_fsf300_s2woocc_generator_mul_lrdlr2e-3_fl2e-2_dsdetach',
  # "expname": 'lego_hybrid/31.06_grad_allcode3x_s2g2k_fsf300_s2woocc_generator_mul_lrdlr2e-3_fl2e-2_dsdetach/eval_c2w_ds4_plr2e-3_s1k',
  # "expname": 'lego_hybrid/26.71_grad_allcode3x_s2g2k_fsf300_generator_mul_fl1e-2_dsdetach/eval_c2w_plr2e-3_s201',
 'expname': 'lego_hybrid/1_div3_ns20',
 'logdir': './logs/syntheticstatic',
 'device': 'cuda:0',

 'data_downsample': 1.0,
 'data_dirs': ['/media/public/disk5/lu/data/nerf_synthetic/lego'],
 'contract': False,
 'ndc': False,
 
 # Pose settigs
    "learn_pose": True,
    "learn_R": True,
    "learn_t": True,
    "pose_lr": 8e-2,#1e-2,#1e-2, #5e-4,  #TODO: modify
    "pose_lr_new": 1e-2,#1e-2,
    "pose_lr_end": 1e-5,
    # "t_lr": 1e-2,
    # "porf_lr": 2e-3,

  "switch2grids_step": 2001,#3001,#2001, #FIXME:
  # "f_step": 300, #FIXME:300, 
  "field_lr": [2e-2, 2e-2, 1e-2],#[1e-2, 5e-3, 1e-3],#[1e-2, 5e-3, 1e-3, 5e-4],
  # 'field_lr': [15e-3, 10e-3, 5e-3],
  "lr": 5e-3,#0.01
  "attn_lr": 1e-2,# 5e-3
  "density_lr": 5e-3,#3e-3,#3e-3,#5e-3,
  "eval_pose_lr": 2e-3,#1e-3,#2e-3,
 'camera_noise': 0.2,#0.25,
  
 # Optimization settings
  "eval_steps": 1001,#10_01,
  "num_steps": 60_001, #30_001,
  "batch_size": 8192, #4096,
  "eval_batch_size": 4096, #4096,
  "code_mode": "mul",
  "code_mode_ds": "mul",
  "ms_res": False,
  "scheduler_type": "warmup_cosine",
  "warmup_step": 512,
  'scheduler_type_pose': "warmup_expdecay",#"warmup_step",
  'scheduler_type_pose_s2': "expdecay",#"warmup_expdecay",
  "eval_scheduler_type": "step",
  "pose_warmup_steps": 256,#100,
  "optim_type": "adam",
  
 # Regularization
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'histogram_loss_weight': 1.0,
 'distortion_loss_weight': 0.001,

 # Training settings
 'save_every': 30000,
 'valid_every': 30000,
 'render_every': -1,#200,#1000,
 'save_outputs': True,
 'train_fp16': False, #True,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 # proposal sampling
 'num_proposal_samples': [256, 128],
 'num_proposal_iterations': 2,
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
   {'num_input_coords': 3, 'num_output_coords': 16, 'resolution': [64, 64, 64]},  #'num_output_coords': 8, 
   {'num_input_coords': 3, 'num_output_coords': 16, 'resolution': [128, 128, 128]}
 ],
  'proposal_generator_args_list': [
      {'model_ch': 8, 'model_out_ch': 16, 'model_res': [32, 32, 32], 'up_block_num': 3, 'block_out_channels': (16,32,64)},
      {'model_ch': 8, 'model_out_ch': 16, 'model_res': [64, 64, 64], 'up_block_num': 3, 'block_out_channels': (16,32,64)}
  ],

 # Model settings
  "tensor_config": ['xy', 'xz', 'yz'],#['xy', 'yz', 'zx'], @remind 和kplanes grids的顺序一致
  "model_ch": 16,#8,
  "model_out_ch": 32*3,#16,
  "model_out_ch_": 32*3,#16,
  "model_res": [20,20,20],  # [40,40,40]
  'model_up_block_num': 4,#5,
  'model_block_out_channels': (64, 64, 128, 256), #(32, 64, 64, 128, 256),
    
    "pose_ch": 8,
    "pose_out_ch": 16,
    "pose_res": [20,20,20],
    'pose_up_block_num': 3,
    'pose_block_out_channels': (32, 64, 128), #(32, 64, 64, 128, 256),
   
 'multiscale_res': [1, 2, 4],  # [1, 2, 4]
 'density_activation': 'trunc_exp',
 'concat_features_across_scales': True,
 'linear_decoder': False,
 'grid_config': [{
   'grid_dimensions': 2,
   'input_coordinate_dim': 3,
   'output_coordinate_dim': 32,
   'resolution': [128, 128, 128]
 }],
}
