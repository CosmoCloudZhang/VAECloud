import os
import time
import h5py
import numpy
import pickle
import argparse
from sklearn import decomposition


def main(tag, name, type, label, folder, directory):
    '''
    Main function for the PCA analysis
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the galaxy sample
        type (str): The type of the configuration
        label (str): The label of the configuration
        folder (str): The folder containing the datasets
        directory (str): The directory containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    pca_folder = os.path.join(folder, '{}/{}/PCA/'.format(name, tag))
    ensemble_folder = os.path.join(folder, '{}/{}/ENSEMBLE/'.format(name, tag))
    
    os.makedirs(os.path.join(folder, '{}/'.format(name)), exist_ok = True)
    os.makedirs(os.path.join(folder, '{}/{}/'.format(name, tag)), exist_ok = True)
    
    os.makedirs(pca_folder, exist_ok = True)
    os.makedirs(ensemble_folder, exist_ok = True)
    
    # Load 
    with h5py.File(os.path.join(directory, 'SYNTHESIZE/{}/{}_{}.hdf5'.format(tag, type, label)), 'r') as file:
        input_data = file[name.lower()]['data'][...]
        input_average = file[name.lower()]['average'][...]
    
    data_size, bin_size, z_size = input_data.shape
    bin_size, z_size = input_average.shape
    
    # PCA
    pca_size = 128
    pca_model = decomposition.PCA(n_components = pca_size)
    eigenvalue_data = pca_model.fit_transform(numpy.reshape(input_data, (data_size, bin_size * z_size)))
    
    eigenvector_data = numpy.reshape(pca_model.components_, (pca_size, bin_size, z_size))
    output_data = numpy.reshape(pca_model.inverse_transform(eigenvalue_data), (data_size, bin_size, z_size))
    
    # Save
    with h5py.File(os.path.join(ensemble_folder, 'INPUT.hdf5'), 'w') as file:
        file.create_dataset('data', data = input_data)
        file.create_dataset('average', data = input_average)
    
    with open(os.path.join(pca_folder, 'MODEL.pkl'), 'wb') as file:
        pickle.dump(pca_model, file)
    
    with h5py.File(os.path.join(pca_folder, 'DATA.hdf5'), 'w') as file:
        file.create_dataset('output', data = output_data)
        file.create_dataset('eigenvalue', data = eigenvalue_data)
        file.create_dataset('eigenvector', data = eigenvector_data)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='PCA')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the galaxy sample')
    PARSE.add_argument('--type', type=str, required=True, help='The type of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    PARSE.add_argument('--directory', type=str, required=True, help='The directory containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    TYPE = PARSE.parse_args().type
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    DIRECTORY = PARSE.parse_args().directory
    
    # Output
    OUTPUT = main(TAG, NAME, TYPE, LABEL, FOLDER, DIRECTORY)