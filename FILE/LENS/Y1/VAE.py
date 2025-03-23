import os
import time
import h5py
import numpy
import pickle
import random
import argparse
from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow import math, keras
from tensorflow_probability import stats

from keras.regularizers import L2
from keras.initializers import HeNormal
from keras import losses, Model, Sequential
from keras.layers import Layer, Flatten, Reshape, Dense, LeakyReLU, Conv1D, Conv1DTranspose, Dropout, BatchNormalization

@keras.saving.register_keras_serializable()
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
        
        # Use Dense layers to modify the data from (pca_size) to (256)
        self.encoder.add(Dense(
            units=256,
            bias_regularizer=L2(0.01),
            kernel_regularizer=L2(0.01),
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Reshape the data from (256) to (256, 1)
        self.encoder.add(Reshape((-1, 1)))
        
        # Use Conv1D layers to modify the data from (256, 1) to (256, 128)
        self.encoder.add(Conv1D(
            strides=1, 
            filters=128, 
            kernel_size=3, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Use Conv1D layers to modify the data from (256, 128) to (128, 128)
        self.encoder.add(Conv1D(
            strides=2, 
            filters=256, 
            kernel_size=3, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Use Conv1D layers to modify the data from (128, 128) to (64, 128)
        self.encoder.add(Conv1D(
            strides=2, 
            filters=512, 
            kernel_size=3, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.encoder.add(BatchNormalization())
        self.encoder.add(LeakyReLU())
        self.encoder.add(Dropout(0.2))
        
        # Use Conv1D layers to modify the data from (64, 128) to (32, 1024)
        self.encoder.add(Conv1D(
            strides=2, 
            filters=1024, 
            kernel_size=3, 
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
    
    def call(self, input_value):
        '''
        Call function for the encoder.
        
        Arguments:
            input_value (numpy.ndarray): The input data.
        
        Returns:
            latent_mu (numpy.ndarray): The mean of the latent space.
            latent_logvar (numpy.ndarray): The log-variance of the latent space.
        '''
        
        latent_value = self.encoder(input_value)
        
        latent_mu = self.mu(latent_value)
        latent_logvar = self.logvar(latent_value)
        
        return latent_mu, latent_logvar

@keras.saving.register_keras_serializable()
class Decoder(Layer):
    '''
    Decoder class for the VAE model.
    
    Arguments:
        Layer (keras.layers.Layer): The base class for the decoder.
    
    Returns:
        output_value (keras.layers.Conv1DTranspose): The output data.
    '''
    def __init__(self, pca_size):
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
            kernel_size=3, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.decoder.add(BatchNormalization())
        self.decoder.add(LeakyReLU())
        self.decoder.add(Dropout(0.2))
        
        # Use Conv1DTranspose layers to modify the data from (64, 512) to (128, 512)
        self.decoder.add(Conv1DTranspose(
            strides=2, 
            filters=256,
            kernel_size=3, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.decoder.add(BatchNormalization())
        self.decoder.add(LeakyReLU())
        self.decoder.add(Dropout(0.2))
        
        # Use Conv1DTranspose layers to modify the data from (128, 512) to (256, 256)
        self.decoder.add(Conv1DTranspose(
            strides=2, 
            filters=128,
            kernel_size=3, 
            padding='same', 
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        self.decoder.add(BatchNormalization())
        self.decoder.add(LeakyReLU())
        self.decoder.add(Dropout(0.2))
        
        # Use Conv1DTranspose layers to modify the data from (256, 256) to (256, 1)
        self.decoder.add(Conv1DTranspose(
            strides=1, 
            filters=1,
            kernel_size=3, 
            padding='same',
            bias_regularizer=L2(0.01), 
            kernel_regularizer=L2(0.01), 
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
        
        # Flatten the data from (pca_size, 1) to (pca_size)
        self.decoder.add(Flatten()) 
        
        # Use Dense layers to modify the data from (256) to (pca_size)
        self.decoder.add(Dense(
            units=pca_size,
            bias_regularizer=L2(0.01),
            kernel_regularizer=L2(0.01),
            activity_regularizer=L2(0.01),
            kernel_initializer=HeNormal()
        ))
    
    def call(self, latent_value):
        '''
        Call function for the decoder.
        
        Arguments:
            latent_value (numpy.ndarray): The latent data.
        
        Returns:
            output_value (numpy.ndarray): The output data.
        '''
        output_value = self.decoder(latent_value)
        return output_value

@keras.saving.register_keras_serializable()
class VAE(Model):
    '''
    VAE class for the VAE model.
    
    Arguments:
        Model (keras.Model): The base class for the VAE
    
    Returns:
        None
    '''
    def __init__(self, theta, pca_size, latent_size):
        '''
        Initialization function for the VAE
        
        Arguments:
            data_size (tuple): The size of the data.
            latent_size (int): The size of the latent space.
        
        Returns:
            None
        '''
        super(VAE, self).__init__()
        
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(pca_size)
        self.theta = theta
        
        self.all_loss_tracker = keras.metrics.Mean(name='all_loss')
        self.mean_loss_tracker = keras.metrics.Mean(name='mean_loss')
        self.scatter_loss_tracker = keras.metrics.Mean(name='scatter_loss')
        self.covariance_loss_tracker = keras.metrics.Mean(name='covariance_loss')
        self.divergence_loss_tracker = keras.metrics.Mean(name='divergence_loss')
    
    def call(self, input_value):
        '''
        Call function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input data.
        
        Returns:
            output_value (numpy.ndarray): The output data.
        '''
        
        latent_mu, latent_logvar = self.encoder(input_value)
        latent_value = self.sampler(latent_mu, latent_logvar)
        output_value = self.decoder(latent_value)
        
        return output_value
    
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
    
    def mean_loss(self, input_value, output_value):
        '''
        Mean loss function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input data.
            output_value (numpy.ndarray): The output data.
        
        Returns:
            float: The mean loss.
        '''
        input_mean = math.reduce_mean(input_value, axis=0)
        output_mean = math.reduce_mean(output_value, axis=0)
        return math.reduce_mean(losses.mean_squared_error(input_mean, output_mean))
    
    def scatter_loss(self, input_value, output_value):
        '''
        Scatter loss function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input data.
            output_value (numpy.ndarray): The output data.
        '''
        input_scatter = math.reduce_std(input_value, axis=0)
        output_scatter = math.reduce_std(output_value, axis=0)
        return math.reduce_mean(losses.mean_squared_error(input_scatter, output_scatter))
    
    def covariance_loss(self, input_value, output_value):
        '''
        Covariance loss function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input data.
            output_value (numpy.ndarray): The output data.
        
        Returns:
            float: The covariance loss.
        '''
        input_covariance = stats.covariance(input_value, sample_axis=0)
        output_covariance = stats.covariance(output_value, sample_axis=0)
        return math.reduce_mean(losses.mean_absolute_error(input_covariance, output_covariance))
    
    def divergence_loss(self, latent_mu, latent_logvar):
        '''
        Divergence loss function for the VAE
        
        Arguments:
            latent_mu (numpy.ndarray): The mean of the latent space.
            latent_logvar (numpy.ndarray): The log-variance of the latent space.
        
        Returns:
            float: The divergence loss.
        '''
        return math.reduce_mean(- 0.5 * (1 + latent_logvar - math.square(latent_mu) - math.exp(latent_logvar)))
    
    def sampler(self, mu, logvar):
        '''
        Sampler function for the VAE
        
        Arguments:
            mu (numpy.ndarray): The mean of the latent space
            logvar (numpy.ndarray): The log-variance of the latent space
        
        Returns:
            numpy.ndarray: The sampled latent data.
        '''
        epsilon = tensorflow.random.normal(shape=tensorflow.shape(mu))
        return mu + epsilon * math.exp(0.5 * logvar)
    
    def train_step(self, input_value):
        '''
        Train step function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input data
        
        Returns:
            dict: The dictionary of losses.
        '''
        with tensorflow.GradientTape() as tape:
            
            latent_mu, latent_logvar = self.encoder(input_value)
            latent_value = self.sampler(latent_mu, latent_logvar)
            output_value = self.decoder(latent_value)
            
            mean_loss = self.mean_loss(input_value, output_value)
            scatter_loss = self.scatter_loss(input_value, output_value)
            covariance_loss = self.covariance_loss(input_value, output_value)
            divergence_loss = self.divergence_loss(latent_mu, latent_logvar)
            
            alpha, beta, gamma, kappa = self.theta
            all_loss = alpha * mean_loss + beta * scatter_loss + gamma * covariance_loss + kappa * divergence_loss
            
            gradients = tape.gradient(all_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            
            self.all_loss_tracker.update_state(all_loss)
            self.mean_loss_tracker.update_state(mean_loss)
            self.scatter_loss_tracker.update_state(scatter_loss)
            self.covariance_loss_tracker.update_state(covariance_loss)
            self.divergence_loss_tracker.update_state(divergence_loss)
        
        return {'all_loss': self.all_loss_tracker.result(), 'mean_loss': self.mean_loss_tracker.result(), 'scatter_loss': self.scatter_loss_tracker.result(), 'covariance_loss': self.covariance_loss_tracker.result(), 'divergence_loss': self.divergence_loss_tracker.result()}
    
    def test_step(self, validation_value):
        '''
        Test step function for the VAE
        
        Arguments:
            test_value (numpy.ndarray): The test data
        
        Returns:
            dict: The dictionary of losses.
        '''
        input_value, input_value = validation_value
        
        latent_mu, latent_logvar = self.encoder(input_value)
        latent_value = self.sampler(latent_mu, latent_logvar)
        output_value = self.decoder(latent_value)
        
        mean_loss = self.mean_loss(input_value, output_value)
        scatter_loss = self.scatter_loss(input_value, output_value)
        covariance_loss = self.covariance_loss(input_value, output_value)
        divergence_loss = self.divergence_loss(latent_mu, latent_logvar)
        
        alpha, beta, gamma, kappa = self.theta
        all_loss = alpha * mean_loss + beta * scatter_loss + gamma * covariance_loss + kappa * divergence_loss
        
        self.all_loss_tracker.update_state(all_loss)
        self.mean_loss_tracker.update_state(mean_loss)
        self.scatter_loss_tracker.update_state(scatter_loss)
        self.covariance_loss_tracker.update_state(covariance_loss)
        self.divergence_loss_tracker.update_state(divergence_loss)
        
        return {'all_loss': self.all_loss_tracker.result(), 'mean_loss': self.mean_loss_tracker.result(), 'scatter_loss': self.scatter_loss_tracker.result(), 'covariance_loss': self.covariance_loss_tracker.result(), 'divergence_loss': self.divergence_loss_tracker.result()}


def main(tag, name, folder):
    '''
    Main function for the VAE analysis
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the galaxy sample
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    pca_folder = os.path.join(folder, '{}/{}/PCA/'.format(name, tag))
    vae_folder = os.path.join(folder, '{}/{}/VAE/'.format(name, tag))
    
    os.makedirs(vae_folder, exist_ok = True)
    
    # Seed
    seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)
    
    # Load 
    with open(os.path.join(pca_folder, 'MODEL.pkl'), 'rb') as file:
        pca_model = pickle.load(file)
    pca_size = pca_model.n_components_
    
    with h5py.File(os.path.join(pca_folder, 'DATA.hdf5'), 'r') as file:
        input_data = file['input'][...]
        pca_eigenvalue = file['eigenvalue'][...]
    data_size, bin_size, z_size = input_data.shape
    
    # PCA
    pca_scale = numpy.max(numpy.std(pca_eigenvalue, axis=0))
    input_value = pca_eigenvalue / pca_scale
    
    # Hyperparameters
    latent_size = 5
    epoch_size = 1024
    batch_size = 4096
    learning_rate = 2e-5
    theta = (1.0, 10.0, 10.0, 100.0)
    
    # Split
    train_value, evaluation_value = train_test_split(input_value, test_size=0.2)
    validation_value, test_value = train_test_split(evaluation_value, test_size=0.5)
    
    # Device
    devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tensorflow.config.experimental.set_memory_growth(device, True)
    
    # Model
    strategy = tensorflow.distribute.MirroredStrategy()
    with strategy.scope():
        
        # Train
        vae_model = VAE(theta, pca_size, latent_size)
        vae_model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1, clipnorm=1.0))
        history = vae_model.fit(train_value, shuffle=True, verbose=2, batch_size=batch_size, epochs=epoch_size, validation_data=[validation_value, validation_value])
        
        # Test
        test_input_data = numpy.reshape(pca_model.inverse_transform(test_value * pca_scale), (-1, bin_size, z_size))
        test_output_data = numpy.reshape(pca_model.inverse_transform(vae_model.predict(test_value) * pca_scale), (-1, bin_size, z_size))
        
        # Latent
        latent_mu, latent_logvar = vae_model.encoder(test_value)
        latent_value = vae_model.sampler(latent_mu, latent_logvar)
        
        # Save
        vae_model.save(os.path.join(vae_folder, 'MODEL/'))
        with h5py.File(os.path.join(vae_folder, 'DATA.hdf5'), 'w') as file:
            file.create_group('data')
            file['data'].create_dataset('latent', data=latent_value)
            file['data'].create_dataset('input', data=test_input_data)
            file['data'].create_dataset('output', data=test_output_data)
            
            file.create_group('meta')
            file['meta'].create_dataset('z_size', data=z_size)
            file['meta'].create_dataset('pca_size', data=pca_size)
            file['meta'].create_dataset('bin_size', data=bin_size)
            file['meta'].create_dataset('pca_scale', data=pca_scale)
            file['meta'].create_dataset('data_size', data=data_size)
            file['meta'].create_dataset('epoch_size', data=epoch_size)
            file['meta'].create_dataset('batch_size', data=batch_size)
            file['meta'].create_dataset('latent_size', data=latent_size)
            file['meta'].create_dataset('learning_rate', data=learning_rate)
            
            file.create_group('history')
            file['history'].create_dataset('epoch', data=history.epoch)
            file['history'].create_dataset('all_loss', data=history.history['all_loss'])
            file['history'].create_dataset('mean_loss', data=history.history['mean_loss'])
            file['history'].create_dataset('scatter_loss', data=history.history['scatter_loss'])
            file['history'].create_dataset('covariance_loss', data=history.history['covariance_loss'])
            file['history'].create_dataset('divergence_loss', data=history.history['divergence_loss'])
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='VAE')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the galaxy sample')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, FOLDER)
