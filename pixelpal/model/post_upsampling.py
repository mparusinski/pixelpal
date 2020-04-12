from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D

from pixelpal.model.base import AbstractAugmentor


class PostUpsampling(AbstractAugmentor):

    def __init__(self):
        super().__init__()

    def build_model(self, input_shape=(32, 32), channels=4, **kwargs):
        filters = kwargs.get('filters', 64)
        kernel_size = kwargs.get('kernel_size', 3)
        num_convolution_layers = kwargs.get('num_convolution_layers', 2)

        input_layer = Input(shape=(*input_shape, channels))

        previous_layer = input_layer
        for i in range(num_convolution_layers):
            previous_layer = Conv2D(filters, kernel_size, padding="same")(previous_layer)

        upscale = UpSampling2D(size=(2, 2), interpolation='bilinear')(previous_layer)
        reconstruction = Conv2D(channels, 1, padding="same")(upscale)

        self.model = Model(inputs=input_layer, outputs=reconstruction)