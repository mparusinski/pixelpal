from abc import abstractmethod
import importlib
import numpy as np


class AbstractAugmentor(object):

    def __init__(self, **kwargs):
        self.build_model(**kwargs)

    @abstractmethod
    def build_model(self, input_shape=(32, 32), channels=3, **kwargs):
        self.model = None

    def augment(self, images):
        if type(images) == list and len(images) >= 1:
            image_as_batch = np.array(images)
        elif type(images) == list and len(images) == 0:
            raise Exception("Unable to augment empty list")
        elif type(images) == np.ndarray and len(images.shape) == 4:
            image_as_batch = images
        elif type(images) == np.ndarray and len(images.shape) == 3:
            images_as_batch = images.reshape((1, *images.shape))
        else:
            raise Exception("Unable to augment {}".format(images))
        prediction_batch = self.model.predict(images_as_batch)
        return [ np.array(x) for x in prediction_batch.tolist()]


def __get_model_name__(module_name):
    module_name_tokens = module_name.split('_')
    return ''.join(token.capitalize() for token in module_name_tokens)


def get_model(module_name, **kwargs):
    module = importlib.import_module(module_name)
    modules_tokens = module_name.split('.')
    model_class_name = __get_model_name__(modules_tokens[-1])
    model_class = getattr(module, model_class_name)
    return model_class(**kwargs)
