config = {
#  'expname': 'cutbeef_hybrid/video_1framete',
    # 'expname': 'salmon_hybrid/c2f/1frame/wts_base7k_noise15_all1e-2_dnotnorm_smooth1e-4_vqrf',
    'expname': 'salmon_hybrid/porf_base9k_noise10_all1e-2_dsfdetach_griddetach_smooth1e-3_vqrf',
 #'expname': 'flame_salmon_hybrid_2_new',
 'logdir': './logs/realdynamic/c2f_usgrid_plnew',
#  'record': ['models_all', 'models', 'datasets', 'runners'],  # file backup
 'device': 'cuda:0',

 # Run first for 1 step with data_downsample=4 to generate weights for ray importance sampling
 'data_downsample': 4,#2,
#  'data_dirs': ['/media/public/disk4/gaohc/DyNeRF/dynamic_dataset/dynerf/flame_salmon_1'],
 'data_dirs': ['/media/public/disk5/nerf_series_datas/dynerf_data/flame_salmon_1'],
 'contract': False,
 'ndc': True,
 'ndc_far': 2.6,
 'near_scaling': 0.95,
 'isg': False,
 'isg_step': -1,
 'ist_step': 50000,
 'keyframes': False,
 'scene_bbox': [[-3.0, -1.8, -1.2], [3.0, 1.8, 1.2]],

 # Pose settigs
 "learn_pose": True,
 "learn_R": True,
 "learn_t": True,
 "hold_every": 8,
#  "pts_spacetime_detach": True,

 # Optimization settings
 'eval_steps': 31, #1001,
 'num_steps': 90001, 
 'n_frame_per_cam': 300,
 'n_imgs_eval_per_step': 300,
 'eval_batch_size': 4096,
 'batch_size': 8192,#4096,
 'scheduler_type': 'warmup_cosine',
 'scheduler_type_pose': ["warmup_step", 'warmup_cosine'],
 'pose_warmup_steps':512,       #TODO: update
 "eval_scheduler_type": "step",
 'optim_type': 'adam',
#  'spaceonly_lr': 5e-3,#0.01,
#  'spacetime_lr': 1e-2,#0.01,
#  "field_lr": [2e-2, 1e-2, 2e-3, 1e-3],#[1e-3, 5e-4, 1e-4, 5e-5],#[1e-2, 5e-3, 1e-3, 5e-4],
 'scale_step': [9000, 18000, 27000],#[7000, 14000, 21000],#[9000, 18000, 27000],
 "pose_lr": [1e-2, 1e-2], # [t, porf]  #5e-4,
 "lr": 1e-2,#1e-2,#2e-2,
 "eval_pose_lr": 1e-3,

 # Regularization
 'distortion_loss_weight': 0.001,
 'histogram_loss_weight': 1.0,
 'l1_time_planes': 0.0001,
 'l1_time_planes_proposal_net': 0.0001,
 'plane_tv_weight': 0.0002,
 'plane_tv_weight_proposal_net': 0.0002,
 'time_smoothness_weight': 0.001,#0.001,
 'time_smoothness_weight_proposal_net': 1e-05,

 # Training settings
 "render_every": -1,
 'save_every': 30_000, #30000,
 'valid_every': 30_000, #30_000, #30000,
 'validrender_every': -1, #30000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 'num_proposal_samples': [256, 128],
 'num_proposal_iterations': 2,
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 150]},
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [256, 256, 256, 150]}
 ],

 # Model settings
 'concat_features_across_scales': True,
 'density_activation': 'trunc_exp',
 'linear_decoder': False,
 'multiscale_res': [1, 2, 4, 8],
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 4,
  'output_coordinate_dim': 16,
  'resolution': [64, 64, 64, 150]
 }],
}
