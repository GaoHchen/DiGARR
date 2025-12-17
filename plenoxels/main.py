import argparse
import importlib.util
import logging
import os
import pprint
import sys
from typing import List, Dict, Any
import tempfile
os.environ['CXX'] = 'g++'
import numpy as np
import torch
import torch.utils.data
sys.path.append('.')  #@remind
from plenoxels.runners import video_trainer
from plenoxels.runners import phototourism_trainer
from plenoxels.runners import static_trainer
from plenoxels.utils.create_rendering import render_to_path, decompose_space_time
from plenoxels.utils.parse_args import parse_optfloat
# from shutil import copyfile
from shutil import copytree


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers,
                        force=True)


def load_data(model_type: str, data_downsample, data_dirs, validate_only: bool, render_only: bool, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)  # lu: return float(data_downsample)

    if model_type == "video":
        return video_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)
    elif model_type == "phototourism":
        return phototourism_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs
        )
    else:
        return static_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)


def init_trainer(model_type: str, **kwargs):
    if model_type == "video":
        from plenoxels.runners import video_trainer
        return video_trainer.VideoTrainer(**kwargs)
    elif model_type == "phototourism":
        from plenoxels.runners import phototourism_trainer
        return phototourism_trainer.PhototourismTrainer(**kwargs)
    else:
        from plenoxels.runners import static_trainer
        return static_trainer.StaticTrainer(**kwargs)


def save_config(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
        for key in config.keys():
            f.write("%s\t%s\n" % (key, config[key]))


def file_backup(config):  # backup models
    dir_lis = config['record']
    log_dir = os.path.join(config['logdir'], config['expname'])
    log_dir = os.path.join(log_dir, f"plenoxels_backup")
    os.makedirs(log_dir, exist_ok=True)
    copyfile('./plenoxels/main.py', os.path.join(log_dir, f"main.py"))

    for dir_name in dir_lis:
        cur_dir = os.path.join(log_dir, dir_name)
        os.makedirs(cur_dir, exist_ok=True)

        dir_name = os.path.join('./plenoxels', dir_name)
        files = os.listdir(dir_name)
        for f_name in files:
            if f_name[-3:] == '.py':
                copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))


def file_backup_all(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    src = './plenoxels'
    dst = os.path.join(log_dir, f"plenoxels")
    copytree(src, dst, dirs_exist_ok=True)


def codes_backup(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    print(log_dir)
    os.system("cp -r ./plenoxels " + log_dir)

    # record_path = os.path.join(log_dir, f"plenoxels")
    # os.makedirs(record_path, exist_ok=True)
    # os.system("cp -r ./plenoxels " + record_path)


def main():
    setup_logging()

    p = argparse.ArgumentParser(description="")

    p.add_argument('--render-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--spacetime-only', action='store_true')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--model-step', type=int, default=-1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('override', nargs=argparse.REMAINDER)

    args = p.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  #@note wojiade

    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config
    # Process overrides from argparse into config
    # overrides can be passed from the command line as key=value pairs. E.g.
    # python plenoxels/main.py --config-path plenoxels/config/cfg.py max_ts_frames=200
    # note that all values are strings, so code should assume incorrect data-types for anything
    # that's derived from config - and should not a string.
    overrides: List[str] = args.override
    overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}
    config.update(overrides_dict)
    if "keyframes" in config:
        model_type = "video"
    elif "appearance_embedding_dim" in config:
        model_type = "phototourism"
    else:
        model_type = "static"  # llff
    validate_only = args.validate_only
    render_only = args.render_only
    spacetime_only = args.spacetime_only
    if validate_only and render_only:
        raise ValueError("render_only and validate_only are mutually exclusive.")
    if render_only and spacetime_only:
        raise ValueError("render_only and spacetime_only are mutually exclusive.")
    if validate_only and spacetime_only:
        raise ValueError("validate_only and spacetime_only are mutually exclusive.")

    pprint.pprint(config)
    if validate_only or render_only:
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
    else:
        save_config(config)

    data = load_data(model_type, validate_only=validate_only, render_only=render_only or spacetime_only, **config)  # 没有pose信息的(尚未采光线的)
    config.update(data)
    trainer = init_trainer(model_type, **config)
    if args.log_dir is not None:
        if args.model_step > -1:
            checkpoint_model_path = os.path.join(args.log_dir, f"model_step{args.model_step}.pth")
            checkpoint_pose_path = os.path.join(args.log_dir, f"posenet_step{args.model_step}.pth")
        else:
            checkpoint_model_path = os.path.join(args.log_dir, "model.pth")
            checkpoint_pose_path = os.path.join(args.log_dir, "posenet.pth")
        training_needed = not (validate_only or render_only or spacetime_only)
        trainer.load_model(torch.load(checkpoint_model_path), ckpt_path=checkpoint_model_path, training_needed=training_needed)
        trainer.load_posenet(torch.load(checkpoint_pose_path), ckpt_path=checkpoint_pose_path, training_needed=training_needed)

    if validate_only:
        codes_backup(config)
        # file_backup_all(config)
        trainer.validate()
    elif render_only:
        render_to_path(trainer, extra_name="")
    elif spacetime_only:
        decompose_space_time(trainer, extra_name="")
    else:
        # file_backup(config)
        # file_backup_all(config)
        codes_backup(config)
        trainer.train()


if __name__ == "__main__":
    main()
