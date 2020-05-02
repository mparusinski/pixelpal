from tensorflow.keras.callbacks import EarlyStopping

def produce_callback():
    early_stoping = EarlyStopping(monitor='val_ssim_metric', patience=1, verbose=1)
    return early_stoping