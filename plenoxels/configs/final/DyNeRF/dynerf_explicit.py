config = {
 'expname': 'cutbeef_explicit/down4_noisepose_lr5e-3',
 'logdir': './logs/realdynamic',
 'device': 'cuda:0',

 # Run first for 1 step with data_downsample=4 to generate weights for ray importance sampling
 'data_downsample': 4,
 'data_dirs': ['/media/public/disk5/lu/data/dynerf/cut_roasted_beef'],
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
 "pose_lr": 1e-2, #5e-4,


 # Optimization settings
 'eval_steps': 301, #1001,
 'num_steps': 90001,  #120001,
  'eval_batch_size': 4096,
 'batch_size': 8192,
  'n_imgs_eval_per_step': 10,
 'scheduler_type': 'warmup_cosine',
 'scheduler_type_pose': "warmup_step",
  "eval_scheduler_type": "step",
 'optim_type': 'adam',
 'lr': 5e-3, # 0.01,
 "eval_pose_lr": 1e-3,


 # Regularization
 'distortion_loss_weight': 0.001,
 'histogram_loss_weight': 1.0,
 'l1_time_planes': 0.0001,
 'l1_time_planes_proposal_net': 0.0001,
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'time_smoothness_weight': 0.001,
 'time_smoothness_weight_proposal_net': 1e-05,

 # Training settings
 "render_every": -1,
 'save_every': 30000,
 'valid_every': 30000,
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
 'linear_decoder': True,
 'linear_decoder_layers': 1,
 'multiscale_res': [1, 2, 4, 8],
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 4,
  'output_coordinate_dim': 32,
  'resolution': [64, 64, 64, 150]
 }],
}
