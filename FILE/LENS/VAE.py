import os
import time
import json
import h5py
import numpy
import argparse
from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow import math, keras
from tensorflow_probability import stats

from keras.initializers import HeNormal
from keras import layers, losses, Model, Sequential
from keras.regularizers import OrthogonalRegularizer
from keras.layers import Layer, Flatten, Reshape, Dense, LeakyReLU, Conv1D, Conv1DTranspose, Dropout, BatchNormalization

class Encoder(Layer):
    '''
    Encoder class for the VAE model.
    
    Arguments:
        Layer (keras.layers.Layer): The base class for the encoder.
    
    Returns:
        latent_mu (keras.layers.Dense): The mean of the latent space.
        latent_logvar (keras.layers.Dense): The log-variance of the latent space.
    '''
    def __init__(self, latent_size):
        '''
        Initialization function for the encoder.
        
        Arguments:
            latent_size (int): The size of the latent space.
        
        Returns:
            None
        '''
        super(Encoder, self).__init__()
        self.encoder = Sequential()
        
        self.encoder.add(Conv1D(filters = 64, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = HeNormal()))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Dropout(0.25))
        self.encoder.add(LeakyReLU())
        
        self.encoder.add(Conv1D(filters = 128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = HeNormal()))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Dropout(0.25))
        self.encoder.add(LeakyReLU())
        
        self.encoder.add(Conv1D(filters = 256, kernel_size = 3, strides = 3, padding = 'same', kernel_initializer = HeNormal()))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Dropout(0.25))
        self.encoder.add(LeakyReLU())
        
        self.encoder.add(Conv1D(filters = 512, kernel_size = 3, strides = 5, padding = 'same', kernel_initializer = HeNormal()))
        self.encoder.add(BatchNormalization())
        self.encoder.add(Dropout(0.25))
        self.encoder.add(LeakyReLU())
        
        self.encoder.add(Flatten())
        
        self.mu = Dense(units = latent_size, kernel_initializer = HeNormal(), kernel_regularizer = OrthogonalRegularizer(factor = 0.001))
        self.logvar = Dense(units = latent_size, kernel_initializer = HeNormal(), kernel_regularizer = OrthogonalRegularizer(factor = 0.001))
    
    def call(self, input_data):
        '''
        Call function for the encoder.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
        
        Returns:
            latent_mu (numpy.ndarray): The mean of the latent space.
            latent_logvar (numpy.ndarray): The log-variance of the latent space.
        '''
        latent_data = input_data - math.reduce_sum(input_data, axis = 1, keepdims = True)
        latent_data = self.encoder(latent_data)
        
        latent_mu = self.mu(latent_data)
        latent_logvar = self.logvar(latent_data)
        
        return latent_mu, latent_logvar


class Decoder(Layer):
    '''
    Decoder class for the VAE model.
    
    Arguments:
        Layer (keras.layers.Layer): The base class for the decoder.
    
    Returns:
        output_data (keras.layers.Conv1DTranspose): The output data.
    '''
    def __init__(self, data_size):
        '''
        Initialization function for the decoder.
        
        Arguments:
            data_size (tuple): The size of the data.
        
        Returns:
            None
        '''
        super(Decoder, self).__init__()
        grid_size, bin_size = data_size
        self.decoder = Sequential()
        
        self.decoder.add(Dense(units = grid_size // 60 * 512, kernel_initializer = HeNormal(), kernel_regularizer = OrthogonalRegularizer(factor = 0.001)))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Dropout(0.25))
        self.decoder.add(LeakyReLU())
        
        self.decoder.add(Reshape((grid_size // 60, 512)))
        
        self.decoder.add(Conv1DTranspose(filters = 256, kernel_size = 3, strides = 5, padding = 'same', kernel_initializer = HeNormal()))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Dropout(0.25))
        self.decoder.add(LeakyReLU())
        
        self.decoder.add(Conv1DTranspose(filters = 128, kernel_size = 3, strides = 3, padding = 'same', kernel_initializer = HeNormal()))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Dropout(0.25))
        self.decoder.add(LeakyReLU())
        
        self.decoder.add(Conv1DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = 'same', output_padding = 1, kernel_initializer = HeNormal()))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Dropout(0.25))
        self.decoder.add(LeakyReLU())
        
        self.decoder.add(Conv1DTranspose(filters = bin_size, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = HeNormal()))
    
    def call(self, latent_data):
        '''
        Call function for the decoder.
        
        Arguments:
            latent_data (numpy.ndarray): The latent data.
        
        Returns:
            output_data (numpy.ndarray): The output data.
        '''
        output_data = self.decoder(latent_data)
        output_data = output_data - math.reduce_sum(output_data, axis = 1, keepdims = True)
        
        return output_data

class VAE(Model):
    '''
    VAE class for the VAE model.
    
    Arguments:
        Model (keras.Model): The base class for the VAE.
    
    Returns:
        None
    '''
    def __init__(self, data_size, latent_size):
        '''
        Initialization function for the VAE.
        
        Arguments:
            data_size (tuple): The size of the data.
            latent_size (int): The size of the latent space.
        
        Returns:
            None
        '''
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(data_size)
        
        self.all_loss_tracker = keras.metrics.Mean(name = 'all_loss')
        self.mean_loss_tracker = keras.metrics.Mean(name = 'mean_loss')
        self.scatter_loss_tracker = keras.metrics.Mean(name = 'scatter_loss')
        self.covariance_loss_tracker = keras.metrics.Mean(name = 'covariance_loss')
        self.divergence_loss_tracker = keras.metrics.Mean(name = 'divergence_loss')
        
    def call(self, input_data):
        '''
        Call function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
        
        Returns:
            output_data (numpy.ndarray): The output data.
        '''
        latent_mu, latent_logvar = self.encoder(input_data)
        latent_data = self.sampler(latent_mu, latent_logvar)
        output_data = self.decoder(latent_data)
        return output_data
    
    @property
    def metrics(self):
        '''
        Property function for the metrics.
        
        Arguments:
            None
        
        Returns:
            list: The list of metrics.
        '''
        return [self.all_loss_tracker, self.mean_loss_tracker, self.scatter_loss_tracker, self.covariance_loss_tracker, self.divergence_loss_tracker]
    
    def mean_loss(self, input_data, output_data):
        '''
        Mean loss function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
            output_data (numpy.ndarray): The output data.
        
        Returns:
            float: The mean loss.
        '''
        input_mean = math.reduce_mean(input_data, axis = 0)
        output_mean = math.reduce_mean(output_data, axis = 0)
        return math.reduce_mean(losses.huber(input_mean, output_mean, 0.01))
    
    def scatter_loss(self, input_data, output_data):
        '''
        Scatter loss function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
            output_data (numpy.ndarray): The output data.
        '''
        input_scatter = math.reduce_std(input_data, axis = 0)
        output_scatter = math.reduce_std(output_data, axis = 0)
        return math.reduce_mean(losses.huber(input_scatter, output_scatter, 0.01))
    
    def covariance_loss(self, input_data, output_data):
        '''
        Covariance loss function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
            output_data (numpy.ndarray): The output data.
        
        Returns:
            float: The covariance loss.
        '''
        input_covariance = stats.covariance(input_data, sample_axis = 0, event_axis = (1, 2))
        output_covariance = stats.covariance(output_data, sample_axis = 0, event_axis = (1, 2))
        return math.reduce_mean(losses.mean_absolute_error(input_covariance, output_covariance))
    
    def divergence_loss(self, latent_mu, latent_logvar):
        '''
        Divergence loss function for the VAE.
        
        Arguments:
            latent_mu (numpy.ndarray): The mean of the latent space.
            latent_logvar (numpy.ndarray): The log-variance of the latent space.
        
        Returns:
            float: The divergence loss.
        '''
        return math.reduce_mean(- 0.5 * (1 + latent_logvar - math.square(latent_mu) - math.exp(latent_logvar)))
    
    def sampler(self, mu, logvar):
        '''
        Sampler function for the VAE.
        
        Arguments:
            mu (numpy.ndarray): The mean of the latent space.
            logvar (numpy.ndarray): The log-variance of the latent space.
        
        Returns:
            numpy.ndarray: The sampled latent data.
        '''
        epsilon = tensorflow.random.normal(shape = tensorflow.shape(mu))
        return mu + epsilon * math.exp(logvar * 0.5)
    
    def train_step(self, input_data):
        '''
        Train step function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
        
        Returns:
            dict: The dictionary of losses.
        '''
        with tensorflow.GradientTape() as tape:
            
            latent_mu, latent_logvar = self.encoder(input_data)
            latent_data = self.sampler(latent_mu, latent_logvar)
            output_data = self.decoder(latent_data)         
            
            mean_loss = self.mean_loss(input_data, output_data)
            scatter_loss = self.scatter_loss(input_data, output_data)
            covariance_loss = self.covariance_loss(input_data, output_data)
            divergence_loss = self.divergence_loss(latent_mu, latent_logvar)
            all_loss = mean_loss + scatter_loss + covariance_loss + divergence_loss
            
            gradients = tape.gradient(all_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            
            self.all_loss_tracker.update_state(all_loss)
            self.mean_loss_tracker.update_state(mean_loss)
            self.scatter_loss_tracker.update_state(scatter_loss)
            self.covariance_loss_tracker.update_state(covariance_loss)
            self.divergence_loss_tracker.update_state(divergence_loss)
        
        return {'all_loss': self.all_loss_tracker.result(), 'mean_loss': self.mean_loss_tracker.result(), 'scatter_loss': self.scatter_loss_tracker.result(), 'covariance_loss': self.covariance_loss_tracker.result(), 'divergence_loss': self.divergence_loss_tracker.result()}
    
    def test_step(self, test_data):
        '''
        Test step function for the VAE.
        
        Arguments:
            test_data (numpy.ndarray): The test data.
        
        Returns:
            dict: The dictionary of losses.
        '''
        input_data, evaluation_data = test_data
        
        latent_mu, latent_logvar = self.encoder(input_data)
        latent_data = self.sampler(latent_mu, latent_logvar)
        output_data = self.decoder(latent_data)
        
        mean_loss = self.mean_loss(input_data, output_data)
        scatter_loss = self.scatter_loss(input_data, output_data)
        covariance_loss = self.covariance_loss(input_data, output_data)
        divergence_loss = self.divergence_loss(latent_mu, latent_logvar)
        all_loss = mean_loss + scatter_loss + covariance_loss + divergence_loss
        
        self.all_loss_tracker.update_state(all_loss)
        self.mean_loss_tracker.update_state(mean_loss)
        self.scatter_loss_tracker.update_state(scatter_loss)
        self.covariance_loss_tracker.update_state(covariance_loss)
        self.divergence_loss_tracker.update_state(divergence_loss)
        
        return {'all_loss': self.all_loss_tracker.result(), 'mean_loss': self.mean_loss_tracker.result(), 'scatter_loss': self.scatter_loss_tracker.result(), 'covariance_loss': self.covariance_loss_tracker.result(), 'divergence_loss': self.divergence_loss_tracker.result()}


def main(path, name):
    '''
    Main function for the PCA analysis.
    
    Arguments:
        path (str): The path to the base folder.
        name (str): The name of the sample datasets.
    
    Returns:
        duration (float): The duration of the PCA analysis
    '''
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    os.makedirs(os.path.join(plot_path, '{}/VAE/'.format(name)), exist_ok = True)
    
    # Data
    data_name = os.path.join(data_path, '{}/INPUT.hdf5'.format(name))
    with h5py.File(data_name, 'r') as file:
        input = file['sample'][:].astype(numpy.float64)
    mock_size, bin_size, grid_size = input.shape
    data_size = (grid_size, bin_size)
    
    # Input
    input_mean = numpy.mean(input, axis = 0)
    input_sigma = numpy.std(input, axis = 0)
    
    scale = numpy.max(input_sigma, axis = 1)
    input_data = input - input_mean[numpy.newaxis, :, :]#numpy.stack([input_mean] * mock_size, axis = 0)
    input_data = numpy.transpose(input_data / scale[numpy.newaxis, :, numpy.newaxis], axes = (0, 2, 1))
    
    # Split
    fraction1 = 0.2 
    fraction2 = 0.5
    train_data, evaluation_data = train_test_split(input_data, test_size = fraction1)
    validation_data, test_data = train_test_split(evaluation_data, test_size = fraction2)
    
    # Device
    devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tensorflow.config.experimental.set_memory_growth(device, True)
    
    # Hyperparameters
    latent_size = 8
    epoch_size = 64
    batch_size = 256
    learning_rate = 0.0001
    
    # Model
    strategy = tensorflow.distribute.MirroredStrategy()
    with strategy.scope():
        model = VAE(data_size, latent_size)
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate, clipvalue = 1.0))
        history = model.fit(train_data, shuffle=True, verbose=2, batch_size=batch_size, epochs=epoch_size, validation_data=[validation_data, validation_data])
    
    # Test
    latent_mu, latent_logvar = model.encoder(test_data)
    latent_data = model.sampler(latent_mu, latent_logvar)
    
    test_input = numpy.transpose(test_data, axes = (0, 2, 1)) * scale[numpy.newaxis, :, numpy.newaxis] + input_mean[numpy.newaxis, :, :]
    test_output = numpy.transpose(model.predict(test_data), axes = (0, 2, 1)) * scale[numpy.newaxis, :, numpy.newaxis] + input_mean[numpy.newaxis, :, :]
    
    # Save
    model.save(os.path.join(data_path, '{}/MODEL'.format(name)))
    
    meta = {
        'BIN_SIZE': bin_size,
        'GRID_SIZE': grid_size,
        'MOCK_SIZE': mock_size,
        'EPOCH_SIZE': epoch_size,
        'BATCH_SIZE': batch_size,
        'LATENT_SIZE': latent_size,
        'LEARNING_RATE': learning_rate
    }
    
    info = {'META': meta, 'HISTORY': history.history}
    with open(os.path.join(data_path, '{}/INFO.json'.format(name)), 'w') as file:
        json.dump(info, file, indent = 4)
    
    with h5py.File(os.path.join(data_path, '{}/DATA.hdf5'.format(name)), 'w') as file:
        file.create_dataset('TEST_INPUT', data = test_input)
        file.create_dataset('TEST_OUTPUT', data = test_output)
        file.create_dataset('LATENT_DATA', data = latent_data)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration

if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB VAE')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the sample datasets')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    
    # Result
    RESULT = main(PATH, NAME)