from tensorflow.keras import Model
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, Conv2DTranspose, AveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


from hdpixels.model.base import ssim_metric, psnr_metric


def up_projection_unit(low_res_prev, nr, kernel_size):
    if type(low_res_prev) == list and len(low_res_prev) >= 2:
        low_res_prev_tilda = Concatenate()(low_res_prev)
        low_res_prev_tilda = Conv2D(nr, 1, padding="same", activation="relu")(low_res_prev_tilda)
    elif type(low_res_prev) == list:
        low_res_prev_tilda = low_res_prev[-1]
    else:
        low_res_prev_tilda = low_res_prev

    high_0 = Conv2DTranspose(nr, kernel_size, padding="same", activation="relu")(low_res_prev_tilda)
    high_0 = UpSampling2D(size=(2,2), interpolation="bilinear")(high_0)

    low_0 = Conv2D(nr, kernel_size, padding="same", activation="relu")(high_0)
    low_0 = AveragePooling2D(pool_size=(2,2), padding="same")(low_0)

    residual = low_0 - low_res_prev_tilda
    high_1 = Conv2DTranspose(nr, kernel_size, padding="same", activation="relu")(residual)
    high_1 = UpSampling2D(size=(2,2), interpolation="bilinear")(high_1)

    sum_layer = high_0 + high_1
    return sum_layer


def down_projection_unit(high_res_prev, nr, kernel_size):
    if type(high_res_prev) == list and len(high_res_prev) >= 2:
        # Problem is somewhere here
        print(high_res_prev)
        high_res_prev_tilda = Concatenate()(high_res_prev)
        high_res_prev_tilda = Conv2D(nr, 1, padding="same", activation="relu")(high_res_prev_tilda)
    elif type(high_res_prev) == list:
        high_res_prev_tilda = high_res_prev[-1]
    else:
        high_res_prev_tilda = high_res_prev

    low_0 = Conv2D(nr, kernel_size, padding="same", activation="relu")(high_res_prev_tilda)
    low_0 = AveragePooling2D(pool_size=(2,2), padding="same")(low_0)

    high_0 = Conv2DTranspose(nr, kernel_size, padding="same", activation="relu")(low_0)
    high_0 = UpSampling2D(size=(2,2), interpolation="bilinear")(high_0)

    residual = high_0 - high_res_prev_tilda
    low_1 = Conv2D(nr, kernel_size, padding="same", activation="relu")(residual)
    low_1 = AveragePooling2D(pool_size=(2,2), padding="same")(low_1)

    sum_layer = low_0 + low_1
    return sum_layer


def initial_feature_extraction(input_layer, n0, nr):
    low_initial = Conv2D(n0, 3, padding="same", activation="relu")(input_layer)    
    low_initial = Conv2D(nr, 1, padding="same", activation="relu")(low_initial)

    return low_initial


def reconstruction(high_res_prev, channels):
    high_res_prev_tilda = Concatenate()(high_res_prev)
    return Conv2D(channels, 3, padding="same", activation="relu")(high_res_prev_tilda)


def create_model(input_shape=(32, 32), channels=4, n0=64, nr=18, kernel_size=6, t=2):
    input_layer = Input(shape=(*input_shape, channels))

    L0 = initial_feature_extraction(input_layer, n0, nr)

    H1 = up_projection_unit(L0, nr, kernel_size)
    L1 = down_projection_unit(H1, nr, kernel_size)

    # I do a weird trick. What I want is to keep the lists of the intermediate H and L layers
    #       [H1, ..., HT] and [L1, ..., LT]
    # I could simply use a H list and a L list which I append at each iteration of the for loop
    # but this results in a "graph disconnected error" (I don't know why). 
    # Using a dictionary which stores each intermediate expected value of the H list and the L list
    # solves everything

    Llists = {0: [L1]}
    Hlists = {0: [H1]}

    for i in range(1, t):
        Hlists[i] = Hlists[i-1] + [up_projection_unit(Llists[i-1], nr, kernel_size)] # [H1, ..., HT]
        Llists[i] = Llists[i-1] + [down_projection_unit(Hlists[i-1], nr, kernel_size)] # [L1, ..., LT]
    print(Hlists)
    Hlists[t] = Hlists[t-1] + [up_projection_unit(Llists[t-1], nr, kernel_size)]

    output_layer = reconstruction(Hlists[t], channels)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[ssim_metric, psnr_metric])

    plot_model(model, to_file='./model.png')
    print(model.summary())

    return model
