from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D

from pixelpal.model.abstract_model import AbstractModelBuilder


class FakeModelBuilder(AbstractModelBuilder):

    def __init__(self):
        super().__init__()

    def create_model(self, input_shape=(32,32), channels=3):
        input_layer = Input(shape=(*input_shape, channels))
        deconv_layer = UpSampling2D(size=(2, 2), interpolation='nearest')(input_layer)
        model = Model(inputs=input_layer, outputs=deconv_layer)

        return model
