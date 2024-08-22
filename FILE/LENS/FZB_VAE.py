import os
import time
import h5py
import numpy
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

import tensorflow
from VAE_MODEL import VAE
from tensorflow import keras
from VAE_FIGURE import plot_loss, plot_data, plot_covariance

def main(path, name, label):
    """
    Main function for the PCA analysis.
    
    Arguments:
        path (str): The path to the base folder.
        name (str): The name of the sample datasets.
        label (str): The label of the sample datasets.
    
    Returns:
        duration (float): The duration of the PCA analysis
    """
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    
    # Data
    data_name = os.path.join(data_path, '{}/{}/INPUT.hdf5'.format(name, label))
    with h5py.File(data_name, 'r') as file:
        input = file['sample'][:].astype(numpy.float32)
    
    mock_size, bin_size, grid_size = input.shape
    data_size = (bin_size, grid_size)
    latent_size = 8
    
    phi0 = 12.0
    phi1 = 0.00
    phi2 = 10.0
    input_data = numpy.transpose(input, axes = (0, 2, 1)) / phi0
    train_data, evaluation_data = train_test_split(input_data, test_size = 0.2)
    validation_data, test_data = train_test_split(evaluation_data, test_size = 0.5)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    rate = 0.0001
    epoch_size = 32
    batch_size = 512
    
    devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tensorflow.config.experimental.set_memory_growth(device, True)
    
    strategy = tensorflow.distribute.MirroredStrategy()
    with strategy.scope():
        model = VAE(latent_size, phi0, data_size, latent_size)
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = rate, clipvalue = 1.0))
    
    history = model.fit(train_data, shuffle = True, verbose = 'auto', epochs = epoch_size, validation_data = [validation_data, validation_data])
    latent_mu, latent_logvar = model.encoder(test_data)
    latent_data = model.sampler(latent_mu, latent_logvar)
    
    test_size = int(mock_size * 0.5 * 0.2)
    test_input = numpy.transpose(test_data, axes = (0, 2, 1)) * phi0
    test_output = numpy.transpose(model.predict(test_data), axes = (0, 2, 1)) * phi0
    
    model.save(os.path.join(data_path, '{}/{}/MODEL/'.format(name, label)))
    numpy.savetxt(os.path.join(data_path, '{}/{}/TEST_INPUT.txt'.format(name, label)), test_data.reshape(test_size, bin_size * grid_size))
    numpy.savetxt(os.path.join(data_path, '{}/{}/TEST_OUTPUT.txt'.format(name, label)), model.predict(test_data).reshape(test_size, bin_size * grid_size))
    
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration