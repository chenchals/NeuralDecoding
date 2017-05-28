import os, json

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..')

CONFIG_PATH = os.path.join(project_root, 'config')
PIPELINE_CONFIG_PATH = os.path.join(CONFIG_PATH, 'pipeline_config.json')



class _pipeline_config(object):
    def __init__(self):
        self._config = json.load(open(PIPELINE_CONFIG_PATH), 'r')
        self.__model_name = self.__get_required('model_name')
        self.__hyper_parameter = self.__get_required('hyper_parameter')

    @property
    def MODEL_NAME(self):
        return self.__model_name

    @MODEL_NAME.setter
    def MODEL_NAME(self, *arg, **kwargs):
        raise Exception("Set property outside the class scope is prohibited")

    @property
    def HYPER_PARAMETER(self):
        return self.__hyper_parameter

    @HYPER_PARAMETER.setter
    def HYPER_PARAMETER(self, *arg, **kwargs):
        raise Exception("Set property outside the class scope is prohibited")

    def __get_required(self, tag):
        ret = self._config.get(tag, None)
        if not ret:
            raise Exception(self.__warn_msg(tag))
        return ret

    def __warn_msg(self, tag):
        return "{0} is not set, please set {0} in config file".format(tag)



PIPELINE_CONFIG = _pipeline_config()