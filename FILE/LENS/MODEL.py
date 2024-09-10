import tensorflow
from tensorflow import math, keras

import tensorflow_probability
from tensorflow_probability import stats

from keras.initializers import HeNormal
from keras import layers, losses, Model, Sequential
from keras.regularizers import OrthogonalRegularizer
from keras.layers import Layer, Flatten, Reshape, Dense, LeakyReLU, Conv1D, Conv1DTranspose, Dropout, BatchNormalization

class Encoder(Layer):
    
    def __init__(self, delta, scale, latent_size):
        
        super(Encoder, self).__init__()
        
        self.delta = delta
        self.scale = scale
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
        
        latent_data = input_data / math.reduce_sum(input_data, axis = 1, keepdims = True) / self.delta / self.scale
        latent_data = self.encoder(latent_data)
        
        latent_mu = self.mu(latent_data)
        latent_logvar = self.logvar(latent_data)
        
        return latent_mu, latent_logvar


class Decoder(Layer):
    
    def __init__(self, delta, scale, data_size, latent_size):
        
        super(Decoder, self).__init__()
        
        grid_size, bin_size = data_size
        self.decoder = Sequential()
        self.delta = delta
        self.scale = scale
        
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
        
        self.decoder.add(Conv1DTranspose(filters = bin_size, kernel_size = 3, strides = 2, padding = 'same', activation = 'softplus', kernel_initializer = HeNormal()))

    def call(self, latent_data):
        
        output_data = self.decoder(latent_data)
        output_data = output_data / math.reduce_sum(output_data, axis = 1, keepdims = True) / self.delta / self.scale
        
        return output_data

class VAE(Model):
    
    def __init__(self, delta, scale, data_size, latent_size):
        
        super(VAE, self).__init__()
        self.encoder = Encoder(delta, scale, latent_size)
        self.decoder = Decoder(delta, scale, data_size, latent_size)
        
        self.all_loss_tracker = keras.metrics.Mean(name = 'all_loss')
        self.mean_loss_tracker = keras.metrics.Mean(name = 'mean_loss')
        self.scatter_loss_tracker = keras.metrics.Mean(name = 'scatter_loss')
        self.covariance_loss_tracker = keras.metrics.Mean(name = 'covariance_loss')
        self.divergence_loss_tracker = keras.metrics.Mean(name = 'divergence_loss')
        
    def call(self, input_data):
        latent_mu, latent_logvar = self.encoder(input_data)
        latent_data = self.sampler(latent_mu, latent_logvar)
        output_data = self.decoder(latent_data)
        return output_data
    
    @property
    def metrics(self):
        return [self.all_loss_tracker, self.mean_loss_tracker, self.scatter_loss_tracker, self.covariance_loss_tracker, self.divergence_loss_tracker]
    
    def mean_loss(self, input_data, output_data):
        
        input_mean = math.reduce_mean(input_data, axis = 0)
        output_mean = math.reduce_mean(output_data, axis = 0)
        return math.reduce_mean(losses.huber(input_mean, output_mean, 0.001))
    
    def scatter_loss(self, input_data, output_data):
        
        input_scatter = math.reduce_std(input_data, axis = 0)
        output_scatter = math.reduce_std(output_data, axis = 0)
        return math.reduce_mean(losses.huber(input_scatter, output_scatter, 0.005))
    
    def covariance_loss(self, input_data, output_data):
        
        input_covariance = stats.covariance(input_data, sample_axis = 0, event_axis = (1, 2))
        output_covariance = stats.covariance(output_data, sample_axis = 0, event_axis = (1, 2))
        return math.reduce_mean(losses.mean_absolute_error(input_covariance, output_covariance))
    
    def divergence_loss(self, latent_mu, latent_logvar):
        return math.reduce_mean(- 0.5 * (1 + latent_logvar - math.square(latent_mu) - math.exp(latent_logvar)))

    def sampler(self, mu, logvar):
        epsilon = tensorflow.random.normal(shape = tensorflow.shape(mu))
        return mu + epsilon * math.exp(logvar * 0.5)
    
    def train_step(self, input_data):
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
