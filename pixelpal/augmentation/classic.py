import numpy as np
from tensorflow.keras.utils import Sequence

class ClassicImageGenerator(Sequence):
    
    def __init__(self, x_data, y_data, batch_size=32, input_dim=(32,32,4), output_dim=(64,64,4), shuffle=True):
        'Initialization'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.x_data = x_data
        self.y_data = y_data
        self.ordered_indices = range(self.x_data.shape[0])
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.ordered_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def __data_generation(self, list_of_indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.input_dim))
        Y = np.empty((self.batch_size, *self.output_dim))
        
        # Generate data
        for i, index in enumerate(list_of_indices):
            # Store sample
            X[i,] = self.x_data[index, :]
            # Store class
            Y[i,] = self.y_data[index, :]
        
        return X, Y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ordered_indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.indices[k] for k in indexes]
        
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        
        return X, Y



def produce_generator():
    return ClassicImageGenerator
