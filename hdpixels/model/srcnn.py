from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D
from tensorflow.keras.optimizers import Adam

from hdpixels.model.base import ssim_metric, psnr_metric


def create_model(input_shape=(32, 32), channels=4, kernel_size_1=9, kernel_size_2=1, kernel_size_3=5, features_1=64, features_2=32, num_conv_layers=1):
    input_layer = Input(shape=(*input_shape, channels))
    
    upscale = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_layer)

    conv_1 = Conv2D(features_1, kernel_size_1, padding="same", activation="relu")(upscale)

    previous_layer = conv_1
    for i in range(num_conv_layers):
        previous_layer = Conv2D(
            features_2, kernel_size_2, padding="same", activation="relu"
        )(previous_layer)

    reconstruction = Conv2D(channels, kernel_size_3, padding="same", activation="sigmoid")(previous_layer)

    model = Model(inputs=input_layer, outputs=reconstruction)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[ssim_metric, psnr_metric])
    print(model.summary())

    return model
