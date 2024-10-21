import os
import time
import h5py
import numpy
import argparse

def main(path, name):
    '''
    Main function for the shift analysis.
    
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
        input = file['sample'][:].astype(numpy.float32)
    
    # Shift
    z1 = 0.0
    z2 = 3.0
    z_size = 300
    z_delta = (z2 - z1) / z_size
    z_grid = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, z_size)
    
    z_mean = numpy.sum(input * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    covariance = numpy.cov(z_mean, rowvar=False, bias=False)
    
    # Save
    numpy.savetxt(os.path.join(data_path, '{}/SHIFT.txt'.format(name)), covariance)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='PCA')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the sample datasets')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    RESULT = main(PATH, NAME)