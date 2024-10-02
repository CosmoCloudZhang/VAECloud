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

from keras.regularizers import L2
from keras.initializers import HeNormal
from keras import losses, Model, Sequential
from keras.layers import Layer, Flatten, Permute, Reshape, Dense, LeakyReLU, Conv1D, Conv1DTranspose, Dropout, BatchNormalization

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
        
        # Use a Dense layer to reshape from (bin_size, grid_size) to (bin_size, 256)
        self.encoder.add(Dense(units=256, kernel_initializer=HeNormal()))
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Reshape the data from (bin_size, 256) to (256, bin_size)
        self.encoder.add(Permute((2, 1)))
        
        # Use Conv1D layers to modify the data from (256, bin_size) to (256, 128)
        self.encoder.add(Conv1D(
            strides=1, 
            filters=128, 
            kernel_size=2, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Use Conv1D layers to modify the data from (256, 128) to (128, 256)
        self.encoder.add(Conv1D(
            strides=2, 
            filters=256, 
            kernel_size=2, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Use Conv1D layers to modify the data from (128, 256) to (64, 512)
        self.encoder.add(Conv1D(
            strides=2, 
            filters=512, 
            kernel_size=2, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Use Conv1D layers to modify the data from (64, 512) to (32, 1024)
        self.encoder.add(Conv1D(
            strides=2, 
            filters=1024, 
            kernel_size=2, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Flatten the data from (32, 1024) to (32 * 1024)
        self.encoder.add(Flatten())
        
        # Use Dense layers to modify the data from (32 * 1024) to (latent_size)
        self.mu = Dense(
            units=latent_size, 
            bias_regularizer=L2(0.01),
            kernel_regularizer=L2(0.01),
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        )
        
        self.logvar = Dense(
            units=latent_size, 
            bias_regularizer=L2(0.01),
            kernel_regularizer=L2(0.01),
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        )
    
    def call(self, input_data):
        '''
        Call function for the encoder.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
        
        Returns:
            latent_mu (numpy.ndarray): The mean of the latent space.
            latent_logvar (numpy.ndarray): The log-variance of the latent space.
        '''
        
        latent_data = self.encoder(input_data)
        
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
    def __init__(self, bin_size, grid_size):
        '''
        Initialization function for the decoder.
        
        Arguments:
            data_size (tuple): The size of the data.
        
        Returns:
            None
        '''
        super(Decoder, self).__init__()
        self.decoder = Sequential()
        
        # Use Dense layers to modify the data from (latent_size) to (32 * 1024)
        self.decoder.add(Dense(
            units=32 * 1024, 
            bias_regularizer=L2(0.01),
            kernel_regularizer=L2(0.01),
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.decoder.add(BatchNormalization())
        self.decoder.add(LeakyReLU())
        self.decoder.add(Dropout(0.2))
        
        # Reshape the data from (32 * 1024) to (32, 1024)
        self.decoder.add(Reshape((32, 1024)))
        
        # Use Conv1DTranspose layers to modify the data from (32, 1024) to (64, 512)
        self.decoder.add(Conv1DTranspose(
            strides=2, 
            filters=512,
            kernel_size=2, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.decoder.add(BatchNormalization())
        self.decoder.add(LeakyReLU())
        self.decoder.add(Dropout(0.2))
        
        # Use Conv1DTranspose layers to modify the data from (64, 512) to (128, 256)
        self.decoder.add(Conv1DTranspose(
            strides=2, 
            filters=256,
            kernel_size=2, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.decoder.add(BatchNormalization())
        self.decoder.add(LeakyReLU())
        self.decoder.add(Dropout(0.2))
        
        # Use Conv1DTranspose layers to modify the data from (128, 256) to (256, 128)
        self.decoder.add(Conv1DTranspose(
            strides=2, 
            filters=128,
            kernel_size=2, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.decoder.add(BatchNormalization())
        self.decoder.add(LeakyReLU())
        self.decoder.add(Dropout(0.2))
        
        # Use Conv1DTranspose layers to modify the data from (256, 128) to (256, bin_size)
        self.decoder.add(Conv1DTranspose(
            strides=1, 
            kernel_size=2, 
            padding='same', 
            filters=bin_size,
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.decoder.add(BatchNormalization())
        self.decoder.add(LeakyReLU())
        self.decoder.add(Dropout(0.2))
        
        # Reshape the data from (256, bin_size) to (bin_size, 256)
        self.decoder.add(Permute((2, 1)))
        
        # Use Dense layers to modify the data from (bin_size, 256) to (bin_size, grid_size)
        self.decoder.add(Dense(
            units=grid_size, 
            activation='softmax',
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
    
    def call(self, latent_data):
        '''
        Call function for the decoder.
        
        Arguments:
            latent_data (numpy.ndarray): The latent data.
        
        Returns:
            output_data (numpy.ndarray): The output data.
        '''
        output_data = self.decoder(latent_data)
        return output_data

class VAE(Model):
    '''
    VAE class for the VAE model.
    
    Arguments:
        Model (keras.Model): The base class for the VAE.
    
    Returns:
        None
    '''
    def __init__(self, alpha, delta, bin_size, grid_size, latent_size):
        '''
        Initialization function for the VAE.
        
        Arguments:
            data_size (tuple): The size of the data.
            latent_size (int): The size of the latent space.
        
        Returns:
            None
        '''
        super(VAE, self).__init__()
        
        self.alpha = alpha
        self.delta = delta
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(bin_size, grid_size)
        
        self.all_loss_tracker = keras.metrics.Mean(name='all_loss')
        self.mean_loss_tracker = keras.metrics.Mean(name='mean_loss')
        self.scatter_loss_tracker = keras.metrics.Mean(name='scatter_loss')
        self.covariance_loss_tracker = keras.metrics.Mean(name='covariance_loss')
        self.divergence_loss_tracker = keras.metrics.Mean(name='divergence_loss')
        
    def call(self, input_data):
        '''
        Call function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
        
        Returns:
            output_data (numpy.ndarray): The output data.
        '''
        
        scale = math.reduce_max(input_data, axis=(0, 2), keepdims=True)
        latent_mu, latent_logvar = self.encoder(input_data / scale)
        latent_data = self.sampler(latent_mu, latent_logvar)
        
        output_data = self.decoder(latent_data) * scale
        output_data = math.divide_no_nan(output_data, math.reduce_sum(output_data, axis=2, keepdims=True)) / self.delta
        
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
        input_mean = math.reduce_mean(input_data, axis=0)
        output_mean = math.reduce_mean(output_data, axis=0)
        return math.reduce_mean(losses.mean_squared_error(input_mean, output_mean))
    
    def scatter_loss(self, input_data, output_data):
        '''
        Scatter loss function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
            output_data (numpy.ndarray): The output data.
        '''
        input_scatter = math.reduce_std(input_data, axis=0)
        output_scatter = math.reduce_std(output_data, axis=0)
        return math.reduce_mean(losses.mean_squared_error(input_scatter, output_scatter))
    
    def covariance_loss(self, input_data, output_data):
        '''
        Covariance loss function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
            output_data (numpy.ndarray): The output data.
        
        Returns:
            float: The covariance loss.
        '''
        input_covariance = stats.covariance(input_data, sample_axis=0, event_axis=(1, 2))
        output_covariance = stats.covariance(output_data, sample_axis=0, event_axis=(1, 2))
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
        epsilon = tensorflow.random.normal(shape=tensorflow.shape(mu))
        return mu + epsilon * math.exp(0.5 * logvar)
    
    def train_step(self, input_data):
        '''
        Train step function for the VAE.
        
        Arguments:
            input_data (numpy.ndarray): The input data.
        
        Returns:
            dict: The dictionary of losses.
        '''
        with tensorflow.GradientTape() as tape:
            
            scale = math.reduce_max(input_data, axis=(0, 2), keepdims=True)
            latent_mu, latent_logvar = self.encoder(input_data / scale)
            latent_data = self.sampler(latent_mu, latent_logvar)
            
            output_data = self.decoder(latent_data) * scale
            output_data = math.divide_no_nan(output_data, math.reduce_sum(output_data, axis=2, keepdims=True)) / self.delta
            
            mean_loss = self.mean_loss(input_data, output_data)
            scatter_loss = self.scatter_loss(input_data, output_data)
            covariance_loss = self.covariance_loss(input_data, output_data)
            divergence_loss = self.divergence_loss(latent_mu, latent_logvar)
            
            beta, gamma, kappa = self.alpha
            all_loss = mean_loss + beta * scatter_loss + gamma * covariance_loss + kappa * divergence_loss
            
            gradients = tape.gradient(all_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            
            self.all_loss_tracker.update_state(all_loss)
            self.mean_loss_tracker.update_state(mean_loss)
            self.scatter_loss_tracker.update_state(scatter_loss)
            self.covariance_loss_tracker.update_state(covariance_loss)
            self.divergence_loss_tracker.update_state(divergence_loss)
        
        return {'all_loss': self.all_loss_tracker.result(), 'mean_loss': self.mean_loss_tracker.result(), 'scatter_loss': self.scatter_loss_tracker.result(), 'covariance_loss': self.covariance_loss_tracker.result(), 'divergence_loss': self.divergence_loss_tracker.result()}
    
    def test_step(self, validation_data):
        '''
        Test step function for the VAE.
        
        Arguments:
            test_data (numpy.ndarray): The test data.
        
        Returns:
            dict: The dictionary of losses.
        '''
        input_data, input_data = validation_data
        
        scale = math.reduce_max(input_data, axis=(0, 2), keepdims=True)
        latent_mu, latent_logvar = self.encoder(input_data / scale)
        latent_data = self.sampler(latent_mu, latent_logvar)
        
        output_data = self.decoder(latent_data) * scale
        output_data = math.divide_no_nan(output_data, math.reduce_sum(output_data, axis=2, keepdims=True)) / self.delta
        
        mean_loss = self.mean_loss(input_data, output_data)
        scatter_loss = self.scatter_loss(input_data, output_data)
        covariance_loss = self.covariance_loss(input_data, output_data)
        divergence_loss = self.divergence_loss(latent_mu, latent_logvar)
        
        beta, gamma, kappa = self.alpha
        all_loss = mean_loss + beta * scatter_loss + gamma * covariance_loss + kappa * divergence_loss
        
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
    
    # Data
    data_name = os.path.join(data_path, '{}/INPUT.hdf5'.format(name))
    with h5py.File(data_name, 'r') as file:
        input = file['sample'][:].astype(numpy.float64)
    mock_size, bin_size, grid_size = input.shape
    
    # Input
    delta = 0.01
    input_data = numpy.maximum(input, 0.0)
    input_data = input_data / numpy.sum(input_data, axis=2, keepdims=True) / delta
    del input
    
    # Split
    train_data, evaluation_data = train_test_split(input_data, test_size=0.2)
    validation_data, test_data = train_test_split(evaluation_data, test_size=0.5)
    
    # Device
    devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tensorflow.config.experimental.set_memory_growth(device, True)
    
    # Hyperparameters
    latent_size = 5
    epoch_size = 512
    batch_size = 1024
    learning_rate = 1e-5
    alpha = (10.0, 100.0, 1000.0)
    
    # Model
    strategy = tensorflow.distribute.MirroredStrategy()
    with strategy.scope():
        model = VAE(alpha, delta, bin_size, grid_size, latent_size)
        model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1, clipnorm=1.0))
        history = model.fit(train_data, shuffle=True, verbose=2, batch_size=batch_size, epochs=epoch_size, validation_data=[validation_data, validation_data])
        
        test_output = model.predict(test_data)
        latent_mu, latent_logvar = model.encoder(test_data)
        latent_data = model.sampler(latent_mu, latent_logvar)
        
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
            json.dump(info, file, indent=4)
        
        with h5py.File(os.path.join(data_path, '{}/DATA.hdf5'.format(name)), 'w') as file:
            file.create_dataset('TEST_INPUT', data=test_data)
            file.create_dataset('TEST_OUTPUT', data=test_output)
            file.create_dataset('LATENT_DATA', data=latent_data)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration

if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='VAE')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the sample datasets')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    
    # Result
    RESULT = main(PATH, NAME)
