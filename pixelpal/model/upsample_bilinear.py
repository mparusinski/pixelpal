from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D

from pixelpal.model.base import AbstractAugmentor


class UpsampleBilinear(AbstractAugmentor):

    def __init__(self):
        super().__init__()

    def build_model(self, input_shape=(32, 32), channels=4):
        input_layer = Input(shape=(*input_shape, channels))
        deconv_layer = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_layer)
        self.model = Model(inputs=input_layer, outputs=deconv_layer)

