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
        latent_mu (keras.layers.Dense): The mean of the latent space
        latent_logvar (keras.layers.Dense): The log-variance of the latent space
    '''
    def __init__(self, latent_size):
        '''
        Initialization function for the encoder.
        
        Arguments:
            latent_size (int): The size of the latent space
        
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
            input_value (numpy.ndarray): The input value
        
        Returns:
            latent_mu (numpy.ndarray): The mean of the latent space
            latent_logvar (numpy.ndarray): The log-variance of the latent space
        '''
        
        latent_data = self.encoder(input_value)
        
        latent_mu = self.mu(latent_data)
        latent_logvar = self.logvar(latent_data)
        
        return latent_mu, latent_logvar

@keras.saving.register_keras_serializable()
class Decoder(Layer):
    '''
    Decoder class for the VAE model.
    
    Arguments:
        Layer (keras.layers.Layer): The base class for the decoder.
    
    Returns:
        output_value (keras.layers.Conv1DTranspose): The output value
    '''
    def __init__(self, pca_size):
        '''
        Initialization function for the decoder.
        
        Arguments:
            data_size (tuple): The size of the data
        
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
    
    def call(self, latent_data):
        '''
        Call function for the decoder.
        
        Arguments:
            latent_data (numpy.ndarray): The latent data
        
        Returns:
            output_value (numpy.ndarray): The output value
        '''
        output_value = self.decoder(latent_data)
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
            data_size (tuple): The size of the data
            latent_size (int): The size of the latent space
        
        Returns:
            None
        '''
        super(VAE, self).__init__()
        
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(pca_size)
        self.theta = theta
        
        self.all_loss_tracker = keras.metrics.Mean(name='all_loss')
        self.mean_loss_tracker = keras.metrics.Mean(name='mean_loss')
        self.deviation_loss_tracker = keras.metrics.Mean(name='deviation_loss')
        self.covariance_loss_tracker = keras.metrics.Mean(name='covariance_loss')
        self.discrepancy_loss_tracker = keras.metrics.Mean(name='discrepancy_loss')
        self.regularization_loss_tracker = keras.metrics.Mean(name='regularization_loss')
    
    def call(self, input_value):
        '''
        Call function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input value
        
        Returns:
            output_value (numpy.ndarray): The output value
        '''
        
        latent_mu, latent_logvar = self.encoder(input_value)
        latent_data = self.sampler(latent_mu, latent_logvar)
        output_value = self.decoder(latent_data)
        
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
        return [self.all_loss_tracker, self.mean_loss_tracker, self.deviation_loss_tracker, self.covariance_loss_tracker, self.discrepancy_loss_tracker, self.regularization_loss_tracker]
    
    def mean_loss(self, input_value, output_value):
        '''
        Mean loss function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input value
            output_value (numpy.ndarray): The output value
        
        Returns:
            float: The mean loss
        '''
        input_mean = math.reduce_mean(input_value, axis=0)
        output_mean = math.reduce_mean(output_value, axis=0)
        return math.reduce_mean(losses.mean_squared_error(input_mean, output_mean))
    
    def deviation_loss(self, input_value, output_value):
        '''
        deviation loss function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input value
            output_value (numpy.ndarray): The output value
        '''
        input_deviation = math.reduce_std(input_value, axis=0)
        output_deviation = math.reduce_std(output_value, axis=0)
        return math.reduce_mean(losses.mean_squared_error(input_deviation, output_deviation))
    
    def covariance_loss(self, input_value, output_value):
        '''
        Covariance loss function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input value
            output_value (numpy.ndarray): The output value
        
        Returns:
            float: The covariance loss
        '''
        input_covariance = stats.covariance(input_value, sample_axis=0)
        output_covariance = stats.covariance(output_value, sample_axis=0)
        return math.reduce_mean(losses.mean_absolute_error(input_covariance, output_covariance))
    
    def discrepancy_loss(self, input_value, output_value):
        '''
        Discrepancy loss function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input value
            output_value (numpy.ndarray): The output value
        
        Returns:
            float: The discrepancy loss
        '''
        input_delta = input_value - math.reduce_mean(input_value, axis=0)
        output_delta = output_value - math.reduce_mean(output_value, axis=0)
        
        input_batch_size = tensorflow.cast(tensorflow.shape(input_value)[0], tensorflow.float32)
        output_batch_size = tensorflow.cast(tensorflow.shape(output_value)[0], tensorflow.float32)
        
        input_tensor = tensorflow.einsum('ni, nj, nk -> ijk', input_delta, input_delta, input_delta) / input_batch_size
        output_tensor = tensorflow.einsum('ni, nj, nk -> ijk', output_delta, output_delta, output_delta) / output_batch_size
        
        return math.reduce_mean(losses.mean_absolute_error(input_tensor, output_tensor))
    
    def regularization_loss(self, latent_mu, latent_logvar):
        '''
        Regularization loss function for the VAE
        
        Arguments:
            latent_mu (numpy.ndarray): The mean of the latent space
            latent_logvar (numpy.ndarray): The log-variance of the latent space
        
        Returns:
            float: The regularization loss
        '''
        return math.reduce_mean(- 0.5 * (1 + latent_logvar - math.square(latent_mu) - math.exp(latent_logvar)))
    
    def sampler(self, mu, logvar):
        '''
        Sampler function for the VAE
        
        Arguments:
            mu (numpy.ndarray): The mean of the latent space
            logvar (numpy.ndarray): The log-variance of the latent space
        
        Returns:
            numpy.ndarray: The sampled latent data
        '''
        epsilon = tensorflow.random.normal(shape=tensorflow.shape(mu))
        return mu + epsilon * math.exp(0.5 * logvar)
    
    def train_step(self, input_value):
        '''
        Train step function for the VAE
        
        Arguments:
            input_value (numpy.ndarray): The input value
        
        Returns:
            dict: The dictionary of losses.
        '''
        with tensorflow.GradientTape() as tape:
            
            latent_mu, latent_logvar = self.encoder(input_value)
            latent_data = self.sampler(latent_mu, latent_logvar)
            output_value = self.decoder(latent_data)
            
            mean_loss = self.mean_loss(input_value, output_value)
            deviation_loss = self.deviation_loss(input_value, output_value)
            covariance_loss = self.covariance_loss(input_value, output_value)
            discrepancy_loss = self.discrepancy_loss(input_value, output_value)
            regularization_loss = self.regularization_loss(latent_mu, latent_logvar)
            
            alpha, beta, tau, nu = self.theta
            all_loss = mean_loss + alpha * deviation_loss + beta * covariance_loss + tau * discrepancy_loss + nu * regularization_loss
            
            gradients = tape.gradient(all_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            
            self.all_loss_tracker.update_state(all_loss)
            self.mean_loss_tracker.update_state(mean_loss)
            self.deviation_loss_tracker.update_state(deviation_loss)
            self.covariance_loss_tracker.update_state(covariance_loss)
            self.discrepancy_loss_tracker.update_state(discrepancy_loss)
            self.regularization_loss_tracker.update_state(regularization_loss)
        
        return {'all_loss': self.all_loss_tracker.result(), 'mean_loss': self.mean_loss_tracker.result(), 'deviation_loss': self.deviation_loss_tracker.result(), 'covariance_loss': self.covariance_loss_tracker.result(), 'discrepancy_loss': self.discrepancy_loss_tracker.result(), 'regularization_loss': self.regularization_loss_tracker.result()}
    
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
        latent_data = self.sampler(latent_mu, latent_logvar)
        output_value = self.decoder(latent_data)
        
        mean_loss = self.mean_loss(input_value, output_value)
        deviation_loss = self.deviation_loss(input_value, output_value)
        covariance_loss = self.covariance_loss(input_value, output_value)
        discrepancy_loss = self.discrepancy_loss(latent_mu, latent_logvar)
        regularization_loss = self.regularization_loss(latent_mu, latent_logvar)
        
        alpha, beta, tau, nu = self.theta
        all_loss = mean_loss + alpha * deviation_loss + beta * covariance_loss + tau * discrepancy_loss + nu * regularization_loss
        
        self.all_loss_tracker.update_state(all_loss)
        self.mean_loss_tracker.update_state(mean_loss)
        self.deviation_loss_tracker.update_state(deviation_loss)
        self.covariance_loss_tracker.update_state(covariance_loss)
        self.discrepancy_loss_tracker.update_state(discrepancy_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)
        
        return {'all_loss': self.all_loss_tracker.result(), 'mean_loss': self.mean_loss_tracker.result(), 'deviation_loss': self.deviation_loss_tracker.result(), 'covariance_loss': self.covariance_loss_tracker.result(), 'discrepancy_loss': self.discrepancy_loss_tracker.result(), 'regularization_loss': self.regularization_loss_tracker.result()}


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
    
    # Folder
    output_folder = os.path.join(folder, '{}/'.format(name))
    
    os.makedirs(output_folder, exist_ok = True)
    os.makedirs(os.path.join(output_folder, '{}/'.format(tag)), exist_ok = True)
    
    # Seed
    seed = 1
    random.seed(seed)
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)
    
    # Load 
    with open(os.path.join(output_folder, '{}/PCA.pkl'.format(tag)), 'rb') as file:
        pca_model = pickle.load(file)
    pca_size = pca_model.n_components_
    
    with h5py.File(os.path.join(output_folder, '{}/PCA.hdf5'.format(tag)), 'r') as file:
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
    theta = (10.0, 100.0, 500.0, 1000.0)
    
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
        test_input_value = test_value * pca_scale
        test_output_value = vae_model.predict(test_value) * pca_scale
        
        test_input_data = numpy.reshape(pca_model.inverse_transform(test_input_value), (-1, bin_size, z_size))
        test_output_data = numpy.reshape(pca_model.inverse_transform(test_output_value), (-1, bin_size, z_size))
        
        # Latent
        test_latent_mu, test_latent_logvar = vae_model.encoder(test_value)
        test_latent_data = vae_model.sampler(test_latent_mu, test_latent_logvar)
        
        # Save
        vae_model.save(os.path.join(output_folder, '{}/VAE.hdf5'.format(tag)))
        with h5py.File(os.path.join(output_folder, '{}/DATA.hdf5'.format(tag)), 'w') as file:
            file.create_group('meta')
            file['meta'].create_dataset('theta', data=theta)
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
            for key, value in history.history.items():
                file['history'].create_dataset(key, data=value)
            
            file.create_group('data')
            file['data'].create_dataset('input', data=test_input_data)
            file['data'].create_dataset('output', data=test_output_data)
            file['data'].create_dataset('latent', data=test_latent_data)
    
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
