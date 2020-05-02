from tensorflow.keras.callbacks import EarlyStopping

def produce_callback():
    early_stoping = EarlyStopping(monitor='val_ssim_metric', patience=2, verbose=1, mode='max')
    return early_stoping