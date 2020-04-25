from tensorflow.keras.callbacks import EarlyStopping

def produce_callback():
    early_stoping = EarlyStopping(monitor='val_mse', min_delta=1e-3, patience=2, verbose=1)
    return early_stoping