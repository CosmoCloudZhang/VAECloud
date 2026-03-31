import os
import time
import h5py
import numpy
import pickle
import argparse
from tensorflow import keras
from matplotlib import cm, colors, pyplot, ticker


def main(tag, folder):
    '''
    Plot the variable of the VAE model
    
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
    
    os.makedirs(os.path.join(figure_folder, 'VARIABLE/'), exist_ok = True)
    
    # Lens
    vae_lens_model = keras.models.load_model(os.path.join(vae_lens_folder, 'MODEL'))
    with open(os.path.join(pca_lens_folder, 'MODEL.pkl'), 'rb') as file:
        pca_lens_model = pickle.load(file)
    
    with h5py.File(os.path.join(vae_lens_folder, 'DATA.hdf5'), 'r') as file:
        z_size = int(file['meta']['z_size'][...])
        bin_lens_size = int(file['meta']['bin_size'][...])
        latent_lens_size = int(file['meta']['latent_size'][...])
    
    # Source
    vae_source_model = keras.models.load_model(os.path.join(vae_source_folder, 'MODEL'))
    with open(os.path.join(pca_source_folder, 'MODEL.pkl'), 'rb') as file:
        pca_source_model = pickle.load(file)
    
    with h5py.File(os.path.join(vae_source_folder, 'DATA.hdf5'), 'r') as file:
        z_size = int(file['meta']['z_size'][...])
        bin_source_size = int(file['meta']['bin_size'][...])
        latent_source_size = int(file['meta']['latent_size'][...])
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, z_size)
    
    # Latent
    latent1 = -5.0
    latent2 = +5.0
    sample_size = 10
    latent_point = numpy.zeros(sample_size + 1)
    latent_sample = numpy.linspace(latent1, latent2, sample_size + 1)
    
    # Figure
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Lens
    figure, plot = pyplot.subplots(nrows = latent_lens_size, ncols = 1, figsize = (15, 5 * latent_lens_size))
    
    for i in range(latent_lens_size):
        
        latent_data = numpy.zeros((sample_size + 1, latent_lens_size))
        for j in range(latent_lens_size):
            if j == i:
                latent_data[:, j] = latent_sample
            else:
                latent_data[:, j] = latent_point
        data_lens = numpy.reshape(pca_lens_model.inverse_transform((vae_lens_model.decoder(latent_data))), (-1, bin_lens_size, z_size))
        
        norm = colors.Normalize(vmin=latent1, vmax=latent2)
        scalar_mappable = cm.ScalarMappable(cmap=cm.Spectral, norm=norm)
        color_list = cm.Spectral(numpy.linspace(0.0, 1.0, sample_size + 1))
        
        for n in range(sample_size + 1):
            for m in range(bin_lens_size):
                plot[i].plot(z_grid, data_lens[n, m, :], color = color_list[n], linewidth = 1.0)
        
        plot[i].set_xlim(0.0, 1.6)
        plot[i].set_ylim(0.0, 8.0)
        
        plot[i].set_xlabel(r'$z$')
        plot[i].set_ylabel(r'$\phi^m (z)$')
        
        plot[i].yaxis.set_major_locator(ticker.FixedLocator(numpy.arange(0.0, 8.0, 1.0)))
        if i < latent_lens_size - 1:
            plot[i].set_xticklabels([])
        
        figure.colorbar(mappable=scalar_mappable, ax=plot[i], orientation='vertical', label=r'$\xi_{}$'.format(i + 1), shrink=0.8)
    figure.subplots_adjust(wspace = 0.05, hspace = 0.00)
    
    figure.savefig(os.path.join(figure_folder, 'VARIABLE/FIGURE_LENS.pdf'), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Source
    figure, plot = pyplot.subplots(nrows = latent_source_size, ncols = 1, figsize = (15, 5 * latent_source_size))
    
    for i in range(latent_source_size):
        
        latent_data = numpy.zeros((sample_size + 1, latent_source_size))
        for j in range(latent_source_size):
            if j == i:
                latent_data[:, j] = latent_sample
            else:
                latent_data[:, j] = latent_point
        data_source = numpy.reshape(pca_source_model.inverse_transform((vae_source_model.decoder(latent_data))), (-1, bin_source_size, z_size))
        
        norm = colors.Normalize(vmin=latent1, vmax=latent2)
        scalar_mappable = cm.ScalarMappable(cmap=cm.Spectral, norm=norm)
        color_list = cm.Spectral(numpy.linspace(0.0, 1.0, sample_size + 1))
        
        for n in range(sample_size + 1):
            for m in range(bin_source_size):
                plot[i].plot(z_grid, data_source[n, m, :], color = color_list[n], linewidth = 1.0)
        
        plot[i].set_xlim(0.0, 1.6)
        plot[i].set_ylim(0.0, 8.0)
        
        plot[i].set_xlabel(r'$z$')
        plot[i].set_ylabel(r'$\phi^m (z)$')
        
        plot[i].yaxis.set_major_locator(ticker.FixedLocator(numpy.arange(0.0, 8.0, 1.0)))
        if i < latent_source_size - 1:
            plot[i].set_xticklabels([])
        
        figure.colorbar(mappable=scalar_mappable, ax=plot[i], orientation='vertical', label=r'$\xi_{}$'.format(i + 1), shrink=0.8)
    figure.subplots_adjust(wspace = 0.05, hspace = 0.00)
    
    figure.savefig(os.path.join(figure_folder, 'VARIABLE/FIGURE_SOURCE.pdf'), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Variable')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)