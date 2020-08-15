from tensorflow.keras.callbacks import ModelCheckpoint

import os

def produce_callback():
    checkpoint_filepath = "./run/weights-improvement-{epoch:02d}-{val_psnr_metric:.2f}-{val_ssim_metric:.2f}.hdf5"
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    return model_checkpoint
