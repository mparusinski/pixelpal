from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

from pixelpal.model.base import ssim_metric, psnr_metric


def create_model(input_shape=(32, 32), channels=4, d=56, s=12, m=4):
    input_layer = Input(shape=(*input_shape, channels))

    feat_conv = Conv2D(d, 5, padding="same", activation="relu")(input_layer)
    shrink_conv = Conv2D(s, 1, padding="same", activation="relu")(feat_conv)

    previous_layer = shrink_conv
    for i in range(m):
        previous_layer = Conv2D(s, 3, padding="same", activation="relu")(previous_layer)

    exp_conv = Conv2D(d, 1, padding="same", activation="relu")(previous_layer)
    upscale = Conv2DTranspose(channels, 9, strides=(2,2), padding="same", activation="sigmoid")(exp_conv)

    model = Model(inputs=input_layer, outputs=upscale)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[ssim_metric, psnr_metric])
    print(model.summary())

    return model
