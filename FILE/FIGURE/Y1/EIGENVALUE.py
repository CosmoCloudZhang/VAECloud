import os
import h5py
import time
import numpy
import pickle
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the eigenvalues of the PCA components
    
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
    vae_lens_folder = os.path.join(folder, 'LENS/{}/VAE/'.format(tag))
    pca_source_folder = os.path.join(folder, 'SOURCE/{}/PCA/'.format(tag))
    vae_source_folder = os.path.join(folder, 'SOURCE/{}/VAE/'.format(tag))
    os.makedirs(os.path.join(figure_folder, 'EIGENVALUE/'), exist_ok = True)
    
    # Load
    with open(os.path.join(pca_lens_folder, 'MODEL.pkl'), 'rb') as file:
        pca_lens_model = pickle.load(file)
    pca_lens_size = pca_lens_model.n_components_
    
    with h5py.File(os.path.join(vae_lens_folder, 'DATA.hdf5'), 'r') as file:
        input_pca_lens_eigenvalue = file['eigenvalue']['input'][...]
        output_pca_lens_eigenvalue = file['eigenvalue']['output'][...]
    
    input_pca_lens_sigma = numpy.std(input_pca_lens_eigenvalue, axis = 0)
    output_pca_lens_sigma = numpy.std(output_pca_lens_eigenvalue, axis = 0)
    vae_lens_sigma = numpy.average([input_pca_lens_sigma, output_pca_lens_sigma], axis = 0)
    
    with h5py.File(os.path.join(vae_source_folder, 'DATA.hdf5'), 'r') as file:
        input_pca_source_eigenvalue = file['eigenvalue']['input'][...]
        output_pca_source_eigenvalue = file['eigenvalue']['output'][...]
    
    input_pca_source_sigma = numpy.std(input_pca_source_eigenvalue, axis = 0)
    output_pca_source_sigma = numpy.std(output_pca_source_eigenvalue, axis = 0)
    vae_source_sigma = numpy.average([input_pca_source_sigma, output_pca_source_sigma], axis = 0)
    
    with open(os.path.join(pca_source_folder, 'MODEL.pkl'), 'rb') as file:
        pca_source_model = pickle.load(file)
    pca_source_size = pca_source_model.n_components_
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Lens
    for n in range(pca_lens_size):
        figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
        
        plot.hist(input_pca_lens_eigenvalue[:, n], range = (- 5 * vae_lens_sigma[n], 5 * vae_lens_sigma[n]), bins = 100, density = False, color = 'darkblue', alpha = 0.5)
        
        plot.hist(output_pca_lens_eigenvalue[:, n], range = (- 5 * vae_lens_sigma[n], 5 * vae_lens_sigma[n]), bins = 100, density = False, color = 'darkred', alpha = 0.5)
        
        plot.set_xlim(- 5 * vae_lens_sigma[n], 5 * vae_lens_sigma[n])
        
        figure.savefig(os.path.join(figure_folder, 'EIGENVALUE/FIGURE_LENS{}.pdf'.format(n + 1)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
    
    # Source
    for n in range(pca_source_size):
        figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
        
        plot.hist(input_pca_source_eigenvalue[:, n], range = (- 5 * vae_source_sigma[n], 5 * vae_source_sigma[n]), bins = 100, density = False, color = 'darkblue', alpha = 0.5)
        
        plot.hist(output_pca_source_eigenvalue[:, n], range = (- 5 * vae_source_sigma[n], 5 * vae_source_sigma[n]), bins = 100, density = False, color = 'darkred', alpha = 0.5)
        
        plot.set_xlim(- 5 * vae_source_sigma[n], 5 * vae_source_sigma[n])
        
        figure.savefig(os.path.join(figure_folder, 'EIGENVALUE/FIGURE_SOURCE{}.pdf'.format(n + 1)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Eigenvalue')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)

