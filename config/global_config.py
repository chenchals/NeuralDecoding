import os, json

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..')

CONFIG_PATH = os.path.join(project_root, 'config')
PIPELINE_CONFIG_PATH = os.path.join(CONFIG_PATH, 'pipeline_config.json')


class PIPELINE_CONFIG:
    _config = json.load(open(PIPELINE_CONFIG_PATH), 'r')
    MODEL_NAME = _config.get('model_name', None)
    if not MODEL_NAME:
        raise Exception("Model name is not set, please set model name in config file")