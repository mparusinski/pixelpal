from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D
from tensorflow.keras.optimizers import Adam

from pixelpal.model.base import ssim_metric, psnr_metric


def create_model(input_shape=(32, 32), channels=4, filters=64, kernel_size=3, num_conv_layers=4):
    input_layer = Input(shape=(*input_shape, channels))

    previous_layer = input_layer
    for i in range(num_conv_layers):
        previous_layer = Conv2D(
            filters, kernel_size, padding="same", activation="relu"
        )(previous_layer)

    upscale = UpSampling2D(size=(2, 2), interpolation='bilinear')(previous_layer)
    reconstruction = Conv2D(channels, 1, padding="same", activation="sigmoid")(upscale)

    model = Model(inputs=input_layer, outputs=reconstruction)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[ssim_metric, psnr_metric])
    print(model.summary())

    return model
