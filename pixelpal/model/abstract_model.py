from abc import abstractmethod


class AbstractModelBuilder(object):

    def __init__(self):
        pass

    @abstractmethod
    def create_model(self, input_shape=(32, 32), channels=3, **kwargs):
        pass
