import os
import time
import numpy
import argparse
import tensorflow
from tensorflow import keras

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
    
    latent_size = 5
    latent_data = numpy.random.normal(size=(1, latent_size))
    
    model = keras.models.load_model(os.path.join(data_path, '{}/MODEL'.format(name)))
    output_data = model.decoder(latent_data)
    print(output_data.shape)
    
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