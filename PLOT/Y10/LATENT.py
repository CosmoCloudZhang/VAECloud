import os
import time
import h5py
import numpy
import corner
import argparse
from matplotlib import pyplot


def main(tag, folder):
    '''
    Plot the latent space of the VAE process
    
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
    vae_lens_folder = os.path.join(folder, 'LENS/{}/VAE/'.format(tag))
    vae_source_folder = os.path.join(folder, 'SOURCE/{}/VAE/'.format(tag))
    os.makedirs(os.path.join(figure_folder, 'LATENT/'), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(vae_lens_folder, 'DATA.hdf5'), 'r') as file:
        latent_lens = file['data']['latent'][...]
        latent_lens_size = int(file['meta']['latent_size'][...])
    
    with h5py.File(os.path.join(vae_source_folder, 'DATA.hdf5'), 'r') as file:
        latent_source = file['data']['latent'][...]
        latent_source_size = int(file['meta']['latent_size'][...])
    
    # Figure
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Lens
    figure = pyplot.figure(figsize=(3 * latent_lens_size, 3 * latent_lens_size))
    
    figure = corner.corner(
        latent_lens,
        figure=figure,
        title_fmt='.1f',
        show_titles=True, 
        quantiles=[0.16, 0.50, 0.84], 
        range=numpy.repeat([(-3.0, 3.0)], latent_lens_size, axis=0),
        labels=[r'$\xi_{}$'.format(k + 1) for k in range(latent_lens_size)]
    )
    
    figure.savefig(os.path.join(figure_folder, 'LATENT/FIGURE_LENS.pdf'), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Source
    figure = pyplot.figure(figsize=(3 * latent_source_size, 3 * latent_source_size))
    
    figure = corner.corner(
        latent_source,
        figure=figure,
        title_fmt='.1f',
        show_titles=True, 
        quantiles=[0.16, 0.50, 0.84], 
        range=numpy.repeat([(-3.0, 3.0)], latent_source_size, axis=0),
        labels=[r'$\xi_{}$'.format(k + 1) for k in range(latent_source_size)]
    )
    
    figure.savefig(os.path.join(figure_folder, 'LATENT/FIGURE_SOURCE.pdf'), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Latent')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)