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
    plot_folder = os.path.join(folder, 'PLOT/')
    lens_folder = os.path.join(folder, 'LENS/')
    source_folder = os.path.join(folder, 'SOURCE/')
    os.makedirs(os.path.join(plot_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(plot_folder, '{}/VARIABLE/'.format(tag)), exist_ok = True)
    
    # Lens
    vae_lens_model = keras.models.load_model(os.path.join(lens_folder, '{}/VAE/'.format(tag)))
    with open(os.path.join(lens_folder, '{}/PCA.pkl'.format(tag)), 'rb') as file:
        pca_lens_model = pickle.load(file)
    
    with h5py.File(os.path.join(lens_folder, '{}/VAE.hdf5'.format(tag)), 'r') as file:
        z_size = int(file['meta']['z_size'][...])
        bin_lens_size = int(file['meta']['bin_size'][...])
        latent_lens_size = int(file['meta']['latent_size'][...])
    
    # Source
    vae_source_model = keras.models.load_model(os.path.join(source_folder, '{}/VAE/'.format(tag)))
    with open(os.path.join(source_folder, '{}/PCA.pkl'.format(tag)), 'rb') as file:
        pca_source_model = pickle.load(file)
    
    with h5py.File(os.path.join(source_folder, '{}/VAE.hdf5'.format(tag)), 'r') as file:
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
    latent_sample = numpy.linspace(latent1, latent2, sample_size + 1)
    
    latent_data_lens = numpy.stack([latent_sample] * latent_lens_size, axis = 1)
    latent_data_source = numpy.stack([latent_sample] * latent_source_size, axis = 1)
    
    data_lens = numpy.reshape(pca_lens_model.inverse_transform((vae_lens_model.decoder(latent_data_lens))), (-1, bin_lens_size, z_size))
    data_source = numpy.reshape(pca_source_model.inverse_transform((vae_source_model.decoder(latent_data_source))), (-1, bin_source_size, z_size))
    
    # Figure
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Lens
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 6))
    
    norm = colors.Normalize(vmin=latent1, vmax=latent2)
    scalar_mappable = cm.ScalarMappable(cmap=cm.plasma_r, norm=norm)
    color_list = cm.plasma_r(numpy.linspace(0.0, 1.0, sample_size + 1))
    
    for n in range(sample_size + 1):
        for m in range(bin_lens_size):
            plot.plot(z_grid, data_lens[n, m, :], color = color_list[n], linewidth = 1.0, rasterized = True)
    
    plot.set_xlim(0.0, 1.4)
    plot.set_ylim(0.0, 12.0)
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\phi^m (z)$')
    plot.yaxis.set_major_locator(ticker.FixedLocator(numpy.arange(0.0, 12.0, 2.0)))
    
    figure.colorbar(mappable=scalar_mappable, ax=plot, orientation='vertical', label=r'$\xi$', shrink=0.9)
    figure.subplots_adjust(wspace = 0.05, hspace = 0.00)
    
    figure.savefig(os.path.join(plot_folder, '{}/VARIABLE/FIGURE_LENS.pdf'.format(tag)), format='pdf', bbox_inches='tight', dpi = 512)
    pyplot.close(figure)
    
    # Source
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 6))
    
    norm = colors.Normalize(vmin=latent1, vmax=latent2)
    scalar_mappable = cm.ScalarMappable(cmap=cm.plasma_r, norm=norm)
    color_list = cm.plasma_r(numpy.linspace(0.0, 1.0, sample_size + 1))
    
    for n in range(sample_size + 1):
        for m in range(bin_source_size):
            plot.plot(z_grid, data_source[n, m, :], color = color_list[n], linewidth = 1.0, rasterized = True)
    
    plot.set_xlim(0.0, 2.1)
    plot.set_ylim(0.0, 6.0)
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\phi^m (z)$')
    plot.yaxis.set_major_locator(ticker.FixedLocator(numpy.arange(0.0, 6.0, 1.0)))
    
    figure.colorbar(mappable=scalar_mappable, ax=plot, orientation='vertical', label=r'$\xi$', shrink=0.9)
    figure.subplots_adjust(wspace = 0.05, hspace = 0.00)
    
    figure.savefig(os.path.join(plot_folder, '{}/VARIABLE/FIGURE_SOURCE.pdf'.format(tag)), format='pdf', bbox_inches='tight', dpi = 512)
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Plot Variable')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder containing the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)
