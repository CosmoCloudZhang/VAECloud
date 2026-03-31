import os
import time
import h5py
import numpy
import pickle
import argparse
from sklearn import decomposition


def main(tag, name, folder):
    '''
    Main function for the PCA analysis
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the galaxy sample
        folder (str): The folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    data_folder = os.path.join(folder, 'DATA/')
    output_folder = os.path.join(folder, '{}/'.format(name))
    
    os.makedirs(output_folder, exist_ok = True)
    os.makedirs(os.path.join(output_folder, '{}/'.format(tag)), exist_ok = True)
    
    # Load 
    with h5py.File(os.path.join(data_folder, '{}/CORRECT.hdf5'.format(tag)), 'r') as file:
        input_data = file[name.lower()]['data'][...]
    data_size, bin_size, z_size = input_data.shape
    
    # PCA
    pca_size = 150
    pca_model = decomposition.PCA(n_components = pca_size)
    eigenvalue_data = pca_model.fit_transform(numpy.reshape(input_data, (data_size, bin_size * z_size)))
    
    eigenvector_data = numpy.reshape(pca_model.components_, (pca_size, bin_size, z_size))
    output_data = numpy.reshape(pca_model.inverse_transform(eigenvalue_data), (data_size, bin_size, z_size))
    
    # Save
    with open(os.path.join(output_folder, '{}/PCA.pkl'.format(tag)), 'wb') as file:
        pickle.dump(pca_model, file)
    
    with h5py.File(os.path.join(output_folder, '{}/PCA.hdf5'.format(tag)), 'w') as file:
        file.create_dataset('input', data = input_data)
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
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, FOLDER)