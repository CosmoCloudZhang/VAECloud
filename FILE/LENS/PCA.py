import os
import time
import h5py
import numpy
import argparse
import PCA_FIGURE
from matplotlib import pyplot
from sklearn import decomposition


def main(path, name):
    """
    Main function for the PCA analysis.
    
    Arguments:
        path (str): The path to the base folder.
        name (str): The name of the sample datasets.
    
    Returns:
        duration (float): The duration of the PCA analysis
    """
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    os.makedirs(os.path.join(plot_path, '{}/PCA/'.format(name)), exist_ok = True)
    
    # Data
    data_name = os.path.join(data_path, '{}/INPUT.hdf5'.format(name))
    with h5py.File(data_name, 'r') as file:
        input = file['sample'][:].astype(numpy.float32)
    
    mock_size, bin_size, grid_size = input.shape
    data_size = bin_size * grid_size
    latent_size = 64
    
    # Model
    input_data = input.reshape(mock_size, data_size)
    model = decomposition.PCA(n_components = latent_size)
    
    latent_data = model.fit_transform(input_data)
    output_data = model.inverse_transform(latent_data)
    output = output_data.reshape(mock_size, bin_size, grid_size)
    
    # Plot
    plot_size = mock_size // 40
    select = numpy.random.choice(numpy.arange(mock_size), plot_size, replace = False)
    
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    figure = PCA_FIGURE.plot_data(input[select, :, :], bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/PCA/INPUT_DATA.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = PCA_FIGURE.plot_data(output[select, :, :], bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/PCA/OUTPUT_DATA.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = PCA_FIGURE.plot_variance(model, latent_size)
    figure.savefig(os.path.join(plot_path, '{}/PCA/VARIANCE.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = PCA_FIGURE.plot_covariance(input_data, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/PCA/INPUT_COVARIANCE.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    figure = PCA_FIGURE.plot_covariance(output_data, bin_size, grid_size)
    figure.savefig(os.path.join(plot_path, '{}/PCA/OUTPUT_COVARIANCE.pdf'.format(name)), bbox_inches = 'tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB PCA')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the sample datasets')
    
    PATH = PARSE.parse_args().path
    NAME = PARSE.parse_args().name
    RESULT = main(PATH, NAME)