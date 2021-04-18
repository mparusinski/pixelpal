from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D
from tensorflow.keras.optimizers import Adam

from pixelpal.model.base import ssim_metric, psnr_metric


def create_model(input_shape=(32, 32), channels=4):
    input_layer = Input(shape=(*input_shape, channels))
    deconv_layer = UpSampling2D(size=(2, 2), interpolation='nearest')(input_layer)

    model = Model(inputs=input_layer, outputs=deconv_layer)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[ssim_metric, psnr_metric])

    return model
