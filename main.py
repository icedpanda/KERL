import argparse
import os.path
import warnings
import shutil

import wandb
import yaml

from src.kerl import check_restore
from src.kerl import run_system

warnings.filterwarnings('ignore')


def initialize_wandb(config, sweep, project_name, group_name, job_type, tags=None):
    if sweep:
        wandb.init(config=config, project=project_name, group=group_name, job_type=job_type, tags=tags,)
        # override config with wandb.config for sweep
        config = wandb.config
        config["sweep"] = True
        if not check_restore(config):
            raise ValueError(
                "restore must be True while using sweep mode to avoid "
                "repeated preprocessing")
    else:
        if config["logger"] == "wandb":
            wandb.login()
        config["sweep"] = False

    return config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, default="config/inspired_kerl.yaml",
        help='config file(yaml) path')
    parser.add_argument(
        '--sweep', default=False, help='whether to sweep hyper parameters',
        dest='sweep', action=argparse.BooleanOptionalAction)
    parser.add_argument(
        '--project_name', type=str, default='kerl', help='wandb project name')
    parser.add_argument(
        '--group_name', type=str, default='base', help='wandb group name')
    parser.add_argument(
        '--job_type', type=str, default='base', help='wandb job type')
    parser.add_argument(
        '--tags', type=str, default='redial', help='wandb tags')

    return parser.parse_args()


def load_config(file_path):
    config = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        config |= yaml.safe_load(f.read())
    return config


if __name__ == '__main__':
    # parse args
    args = parse_arguments()
    # load yaml file to build config dictionary
    config_dict = load_config(args.config)
    config_dict["project_name"] = args.project_name
    config_dict["group_name"] = args.group_name
    config_dict["job_type"] = args.job_type
    config_dict["tags"] = args.tags


    # if pass sweep, update config dictionary
    config_dict = initialize_wandb(config_dict, args.sweep, args.project_name, args.group_name, args.job_type, args.tags)

    run_system(config_dict)
