from tensorflow.keras.callbacks import CSVLogger

import os

def produce_callback():
    csv_filepath = "./run/log.csv"
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    csv_logger = CSVLogger(csv_filepath)
    return csv_logger
