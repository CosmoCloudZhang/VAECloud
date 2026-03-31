import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the eigenvectors of the PCA components
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/{}/'.format(tag))
    pca_lens_folder = os.path.join(folder, 'LENS/{}/PCA/'.format(tag))
    pca_source_folder = os.path.join(folder, 'SOURCE/{}/PCA/'.format(tag))
    os.makedirs(os.path.join(figure_folder, 'EIGENVECTOR/'), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(pca_lens_folder, 'DATA.hdf5'), 'r') as file:
        pca_lens_eigenvector = file['eigenvector'][...]
    pca_lens_size, bin_lens_size, z_size = pca_lens_eigenvector.shape
    
    with h5py.File(os.path.join(pca_source_folder, 'DATA.hdf5'), 'r') as file:
        pca_source_eigenvector = file['eigenvector'][...]
    pca_source_size, bin_source_size, z_size = pca_source_eigenvector.shape
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, z_size)
    
    # Color
    color_lens_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    color_source_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Lens
    for n in range(pca_lens_size):
        figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
        
        for m in range(bin_lens_size):
            plot.plot(z_grid, pca_lens_eigenvector[n, m, :], color = color_lens_list[m], linewidth = 2.5)
        
        plot.set_xlim(z1, z2)
        plot.set_ylim(0.0, numpy.max(pca_lens_eigenvector[n, :, :]) * 1.2)
        
        figure.savefig(os.path.join(figure_folder, 'EIGENVECTOR/FIGURE_LENS{}.pdf'.format(n + 1)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
    
    # Source
    for n in range(pca_source_size):
        figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
        
        for m in range(bin_source_size):
            plot.plot(z_grid, pca_source_eigenvector[n, m, :], color = color_source_list[m], linewidth = 2.5)
        
        plot.set_xlim(z1, z2)
        plot.set_ylim(0.0, numpy.max(pca_source_eigenvector[n, :, :]) * 1.2)
        
        figure.savefig(os.path.join(figure_folder, 'EIGENVECTOR/FIGURE_SOURCE{}.pdf'.format(n + 1)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Eigenvector')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)

