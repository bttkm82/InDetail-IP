import pdb
from configtree import Loader, Walker, Updater
import math
import numpy as np
# from types import SimpleNamespace
from omegaconf import OmegaConf

def walk_configs(config_path):
    walk = None
    update = Updater(namespace={'None': None, 'len': len, 'pi': math.pi, 'log': math.log})
    load = Loader(walk=walk, update=update)
    print(f'walking {config_path}...')
    main_configs = load(config_path)
    main_configs = main_configs.rare_copy() ## to dictionary
    main_configs = OmegaConf.create(main_configs)
    return main_configs

def save_config(save_path, args):
    OmegaConf.save(args, save_path)
# def dict2namespace(dicts):
#     for k, v in dicts.items():
#         if isinstance(v, dict):
#             dicts[k] = dict2namespace(v)
#     return SimpleNamespace(**dicts)